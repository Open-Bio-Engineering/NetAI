"""Training coordinator - orchestrates distributed training across P2P nodes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from typing import Any

import numpy as np

from pydantic import BaseModel, Field

from netai.p2p.network import P2PNode, PeerInfo, NodeState
from netai.training.engine import (
    TrainingConfig,
    TrainingJob,
    TrainingStatus,
    GradientShard,
    LocalTrainer,
    CheckpointManager,
    ShardScheduler,
    GradientCompressor,
    LearningRateScheduler,
)


class NodeContribution(BaseModel):
    node_id: str
    steps_completed: int = 0
    gradients_sent: int = 0
    compute_seconds: float = 0.0
    checkpoints_contributed: int = 0
    last_active: float = 0.0
    device_type: str = "cpu"
    reliability: float = 1.0


logger = logging.getLogger(__name__)


class DistributedTrainingCoordinator:
    def __init__(
        self,
        p2p_node: P2PNode,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.p2p = p2p_node
        self.jobs: dict[str, TrainingJob] = {}
        self.local_trainers: dict[str, LocalTrainer] = {}
        self.shard_scheduler = ShardScheduler()
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir)
        self.contributions: dict[str, dict[str, NodeContribution]] = {}
        self._training_tasks: dict[str, asyncio.Task] = {}
        self._gradient_sync_interval = 5.0
        self._sync_tasks: dict[str, asyncio.Task] = {}
        self.p2p.on("gradient", self._handle_gradient)
        self.p2p.on("training_request", self._handle_training_request)
        self.p2p.on("checkpoint_request", self._handle_checkpoint_request)
        self.p2p.on("model_sync", self._handle_model_sync)
        self.p2p.on("job_status", self._handle_job_status)

    async def submit_job(self, config: TrainingConfig) -> TrainingJob:
        job = TrainingJob(config)
        self.jobs[job.job_id] = job
        self.contributions[job.job_id] = {}
        logger.info("Job %s submitted: %s", job.job_id, config.model_name)
        return job

    async def start_training(self, job_id: str) -> TrainingJob | None:
        job = self.jobs.get(job_id)
        if not job:
            return None
        peers = await self.p2p.peer_table.get_alive_peers()
        self_info = await self.p2p._get_self_info()
        available = [p.model_dump() for p in peers if p.state in (NodeState.ACTIVE, NodeState.TRAINING)]
        available.append(self_info.model_dump())
        assignments = self.shard_scheduler.assign_shards(job, available)
        job.status = TrainingStatus.SCHEDULING
        await self.p2p.broadcast("training_request", {
            "job_id": job.job_id,
            "config": job.config.model_dump(),
            "assignments": {k: v.__dict__ for k, v in assignments.items()},
        })
        my_shard = None
        for sa in assignments.values():
            if sa.node_id == self.p2p.node_id:
                my_shard = sa
                break
        trainer = LocalTrainer(job.config, self.checkpoint_mgr)
        self.local_trainers[job.job_id] = trainer
        task = asyncio.create_task(self._run_local_training(job, trainer, my_shard))
        self._training_tasks[job.job_id] = task
        sync_task = asyncio.create_task(self._gradient_sync_loop(job_id))
        self._sync_tasks[job_id] = sync_task
        return job

    async def _run_local_training(
        self,
        job: TrainingJob,
        trainer: LocalTrainer,
        shard_assignment=None,
    ):
        job.status = TrainingStatus.RUNNING
        job.started_at = time.time()
        layers = shard_assignment.layers if shard_assignment else None
        async def on_step(metrics):
            if layers:
                grad_shard = GradientShard(
                    shard_id=f"shard-{self.p2p.node_id}-{metrics.step}",
                    job_id=job.job_id,
                    step=metrics.step,
                    layer_name=layers[0] if len(layers) == 1 else "all",
                    shape=[1],
                    data=[float(metrics.grad_norm)],
                    num_samples=trainer.config.micro_batch_size,
                    node_id=self.p2p.node_id,
                )
                await job.add_gradient(grad_shard)
                await self.p2p.broadcast("gradient", {
                    "shard_id": grad_shard.shard_id,
                    "job_id": job.job_id,
                    "step": metrics.step,
                    "layer_name": grad_shard.layer_name,
                    "data": grad_shard.data,
                    "node_id": self.p2p.node_id,
                })
            node_id = self.p2p.node_id
            if job.job_id in self.contributions and node_id in self.contributions[job.job_id]:
                c = self.contributions[job.job_id][node_id]
                c.steps_completed += 1
                c.last_active = time.time()
        try:
            await trainer.train(callback=on_step)
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            logger.error("Training job %s failed: %s", job.job_id, e)

    async def _gradient_sync_loop(self, job_id: str):
        while True:
            try:
                await asyncio.sleep(self._gradient_sync_interval)
                job = self.jobs.get(job_id)
                if not job or job.status != TrainingStatus.RUNNING:
                    break
                if job.current_step > 0 and job.current_step % job.config.gradient_accumulation_steps == 0:
                    aggregated = await job.aggregate_gradients(job.current_step)
                    await job.cleanup_old_gradients()
                    if aggregated and job_id in self.local_trainers:
                        trainer = self.local_trainers[job_id]
                        current_state = trainer.get_model_state()
                        for layer, grad in aggregated.items():
                            if layer in current_state:
                                lr = trainer.lr_scheduler.get_lr(job.current_step)
                                current_state[layer] = current_state[layer] - lr * grad
                        trainer.load_model_state(current_state, preserve_optimizer=True)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Gradient sync error: %s", e)

    async def _handle_gradient(self, msg) -> dict:
        data = msg.payload
        job_id = data.get("job_id", "")
        job = self.jobs.get(job_id)
        if not job:
            return {"status": "unknown_job"}
        shard = GradientShard(
            shard_id=data.get("shard_id", ""),
            job_id=job_id,
            step=data.get("step", 0),
            layer_name=data.get("layer_name", ""),
            shape=[1],
            data=data.get("data", []),
            node_id=data.get("node_id", ""),
        )
        await job.add_gradient(shard)
        node_id = data.get("node_id", "")
        if job_id in self.contributions:
            if node_id not in self.contributions[job_id]:
                self.contributions[job_id][node_id] = NodeContribution(node_id=node_id)
            self.contributions[job_id][node_id].gradients_sent += 1
            self.contributions[job_id][node_id].last_active = time.time()
        return {"status": "ok"}

    async def _handle_training_request(self, msg) -> dict:
        data = msg.payload
        config = TrainingConfig(**data.get("config", {}))
        job_id = config.job_id
        if job_id not in self.jobs:
            job = await self.submit_job(config)
        return {"status": "accepted", "job_id": job_id or config.job_id}

    async def _handle_checkpoint_request(self, msg) -> dict:
        data = msg.payload
        job_id = data.get("job_id", "")
        step = data.get("step")
        result = self.checkpoint_mgr.load_checkpoint(job_id, step)
        if not result:
            return {"status": "not_found"}
        state, metadata = result
        return {
            "status": "ok",
            "checkpoint": metadata.model_dump() if metadata else {},
            "model_keys": list(state.keys()),
        }

    async def _handle_model_sync(self, msg) -> dict:
        return {"status": "ok"}

    async def _handle_job_status(self, msg) -> dict:
        data = msg.payload
        job_id = data.get("job_id", "")
        job = self.jobs.get(job_id)
        if not job:
            return {"status": "not_found"}
        m = job.latest_metrics
        return {
            "job_id": job_id,
            "status": job.status.value,
            "step": job.current_step,
            "loss": m.loss if m else 0,
            "best_loss": job.best_loss if job.best_loss != float("inf") else 0,
        }

    async def stop_training(self, job_id: str):
        if job_id in self.local_trainers:
            self.local_trainers[job_id].stop()
        if job_id in self._training_tasks:
            self._training_tasks[job_id].cancel()
        if job_id in self._sync_tasks:
            self._sync_tasks[job_id].cancel()
            del self._sync_tasks[job_id]
        job = self.jobs.get(job_id)
        if job:
            job.status = TrainingStatus.CANCELLED
            job.completed_at = time.time()

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(job_id)
        if not job:
            return None
        m = job.latest_metrics
        contributions = self.contributions.get(job_id, {})
        return {
            "job_id": job_id,
            "status": job.status.value,
            "step": job.current_step,
            "epoch": job.current_epoch,
            "loss": m.loss if m else None,
            "best_loss": job.best_loss if job.best_loss != float("inf") else None,
            "elapsed_s": job.elapsed_seconds,
            "num_checkpoints": len(job.checkpoints),
            "num_contributors": len(contributions),
            "config": job.config.model_dump(),
            "contributions": {n: c.model_dump() for n, c in contributions.items()},
        }

    def list_jobs(self) -> list[dict[str, Any]]:
        return [self.get_job_status(jid) for jid in self.jobs]

    async def export_model(self, job_id: str, output_path: str) -> str:
        job = self.jobs.get(job_id)
        if not job or not self.local_trainers.get(job_id):
            raise ValueError(f"No trainer for job {job_id}")
        trainer = self.local_trainers[job_id]
        state = trainer.get_model_state()
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        np.savez_compressed(output_path, **state)
        return output_path