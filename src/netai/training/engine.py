"""Distributed training engine - gradient sync, shard scheduling, model checkpointing."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import pickle
import shutil
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class TrainingStatus(str, Enum):
    PENDING = "pending"
    SCHEDULING = "scheduling"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    VULKAN = "vulkan"


class OptimizerType(str, Enum):
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    LION = "lion"


class TrainingConfig(BaseModel):
    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_name: str = "gpt2-small"
    model_repo: str = ""
    model_version: str = "main"
    dataset: str = ""
    dataset_config: dict[str, Any] = Field(default_factory=dict)
    optimizer: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    total_steps: int = 10000
    batch_size: int = 8
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    precision: str = "fp32"
    seed: int = 42
    checkpoint_interval: int = 500
    eval_interval: int = 200
    log_interval: int = 10
    device_preference: DeviceType = DeviceType.CUDA
    num_shards: int = 1
    group_id: str = ""
    model_architecture: str = "transformer"
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = Field(default=50257, ge=1)
    intermediate_size: int = Field(default=3072, ge=1)

    @model_validator(mode="after")
    def validate_config(self) -> "TrainingConfig":
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_size <= 0 or self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if self.precision not in ("fp32", "fp16", "bf16", "mixed"):
            raise ValueError(f"precision must be fp16/bf16/fp32/mixed, got {self.precision}")
        return self


class GradientShard(BaseModel):
    shard_id: str
    job_id: str
    step: int
    layer_name: str
    shape: list[int]
    dtype: str = "float32"
    data_hash: str = ""
    data: list[float] = Field(default_factory=list)
    num_samples: int = 0
    node_id: str = ""
    timestamp: float = Field(default_factory=time.time)

    def compute_hash(self) -> str:
        data_hash = hashlib.sha256("".join(str(v) for v in self.data[:100]).encode()).hexdigest()[:16] if self.data else ""
        raw = f"{self.shard_id}:{self.job_id}:{self.step}:{self.layer_name}:{len(self.data)}:{data_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ModelCheckpoint(BaseModel):
    checkpoint_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    job_id: str = ""
    step: int = 0
    epoch: float = 0.0
    loss: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    model_hash: str = ""
    file_path: str = ""
    file_size_mb: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    is_best: bool = False


class TrainingMetrics(BaseModel):
    step: int = 0
    epoch: float = 0.0
    loss: float = 0.0
    lr: float = 0.0
    grad_norm: float = 0.0
    throughput_tok_s: float = 0.0
    gpu_util: float = 0.0
    mem_used_gb: float = 0.0
    elapsed_s: float = 0.0
    samples_seen: int = 0


@dataclass
class ShardAssignment:
    shard_id: str
    node_id: str
    layers: list[str]
    device: DeviceType
    status: str = "assigned"
    steps_completed: int = 0
    last_gradient_step: int = -1


class TrainingJob:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.status = TrainingStatus.PENDING
        self.metrics_history: list[TrainingMetrics] = []
        self.checkpoints: list[ModelCheckpoint] = []
        self.shard_assignments: dict[str, ShardAssignment] = {}
        self.gradient_buffer: dict[int, dict[str, GradientShard]] = {}
        self.best_loss: float = float("inf")
        self.current_step: int = 0
        self.current_epoch: float = 0.0
        self.created_at: float = time.time()
        self.started_at: float = 0.0
        self.completed_at: float = 0.0
        self.error_message: str = ""
        self._gradient_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    @property
    def job_id(self) -> str:
        return self.config.job_id

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at == 0:
            return 0.0
        end = self.completed_at if self.completed_at > 0 else time.time()
        return end - self.started_at

    @property
    def latest_metrics(self) -> TrainingMetrics | None:
        return self.metrics_history[-1] if self.metrics_history else None

    def record_metrics(self, m: TrainingMetrics):
        self.current_step = m.step
        self.current_epoch = m.epoch
        self.metrics_history.append(m)
        if m.loss < self.best_loss:
            self.best_loss = m.loss

    def create_checkpoint(self, step: int, loss: float, metrics: dict[str, float] | None = None) -> ModelCheckpoint:
        ckpt = ModelCheckpoint(
            job_id=self.job_id,
            step=step,
            epoch=self.current_epoch,
            loss=loss,
            metrics=metrics or {},
            is_best=(loss <= self.best_loss),
        )
        self.checkpoints.append(ckpt)
        return ckpt

    async def add_gradient(self, shard: GradientShard):
        async with self._gradient_lock:
            if shard.step not in self.gradient_buffer:
                self.gradient_buffer[shard.step] = {}
            self.gradient_buffer[shard.step][shard.shard_id] = shard
            if len(self.gradient_buffer) > 20:
                steps = sorted(self.gradient_buffer.keys())
                for s in steps[:-10]:
                    del self.gradient_buffer[s]

    async def get_gradients_for_step(self, step: int) -> list[GradientShard]:
        async with self._gradient_lock:
            return list(self.gradient_buffer.get(step, {}).values())

    async def aggregate_gradients(self, step: int) -> dict[str, np.ndarray]:
        shards = await self.get_gradients_for_step(step)
        grouped: dict[str, list[np.ndarray]] = {}
        for shard in shards:
            layer = shard.layer_name
            arr = np.array(shard.data, dtype=np.float32).reshape(shard.shape) if shard.shape else np.array(shard.data, dtype=np.float32)
            grouped.setdefault(layer, []).append(arr)
        result: dict[str, np.ndarray] = {}
        for layer, arrays in grouped.items():
            result[layer] = np.mean(arrays, axis=0)
        return result

    async def cleanup_old_gradients(self, keep_last: int = 5):
        async with self._gradient_lock:
            steps = sorted(self.gradient_buffer.keys())
            if len(steps) > keep_last:
                for s in steps[:-keep_last]:
                    del self.gradient_buffer[s]


class GradientCompressor:
    @staticmethod
    def compress(gradient: np.ndarray, method: str = "topk", ratio: float = 0.01) -> dict[str, Any]:
        flat = gradient.flatten()
        if method == "topk":
            k = max(1, int(len(flat) * ratio))
            top_indices = np.argpartition(np.abs(flat), -k)[-k:]
            return {
                "method": "topk",
                "indices": top_indices.tolist(),
                "values": flat[top_indices].tolist(),
                "shape": gradient.shape,
                "original_size": len(flat),
            }
        elif method == "quantize":
            min_val = float(flat.min())
            max_val = float(flat.max())
            scale = max_val - min_val if max_val != min_val else 1.0
            quantized = ((flat - min_val) / scale * 255).astype(np.uint8)
            return {
                "method": "quantize",
                "quantized": quantized.tolist(),
                "min_val": min_val,
                "scale": scale,
                "shape": gradient.shape,
                "original_size": len(flat),
            }
        return {"method": "none", "values": flat.tolist(), "shape": gradient.shape, "original_size": len(flat)}

    @staticmethod
    def decompress(compressed: dict[str, Any]) -> np.ndarray:
        method = compressed.get("method", "none")
        shape = tuple(compressed.get("shape", (1,)))
        if method == "topk":
            original_size = compressed.get("original_size", 0)
            if original_size == 0 or "indices" not in compressed or "values" not in compressed:
                return np.zeros(shape, dtype=np.float32)
            result = np.zeros(original_size, dtype=np.float32)
            indices = np.array(compressed["indices"])
            values = np.array(compressed["values"])
            result[indices] = values
            return result.reshape(shape)
        elif method == "quantize":
            if "quantized" not in compressed or "scale" not in compressed or "min_val" not in compressed:
                return np.zeros(shape, dtype=np.float32)
            quantized = np.array(compressed["quantized"], dtype=np.uint8)
            result = quantized.astype(np.float32) / 255.0 * compressed["scale"] + compressed["min_val"]
            return result.reshape(shape)
        values = compressed.get("values", [])
        return np.array(values).reshape(shape)


class LearningRateScheduler:
    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int, schedule: str = "cosine"):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule = schedule

    def get_lr(self, step: int) -> float:
        if step < 0:
            step = 0
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / max(1, self.warmup_steps)
        if step >= self.total_steps:
            return self.base_lr * 0.01
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = max(0.0, min(progress, 1.0))
        if self.schedule == "cosine":
            return self.base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
        elif self.schedule == "linear":
            return self.base_lr * (1.0 - progress)
        elif self.schedule == "constant":
            return self.base_lr
        return self.base_lr


class ShardScheduler:
    def __init__(self):
        self.assignments: dict[str, dict[str, ShardAssignment]] = {}

    def assign_shards(
        self,
        job: TrainingJob,
        available_nodes: list[dict[str, Any]],
    ) -> dict[str, ShardAssignment]:
        num_layers = job.config.num_layers
        if not available_nodes:
            num_shards = 1
            available_nodes = [{"node_id": "local", "gpu_count": 0}]
        else:
            num_shards = min(job.config.num_shards, len(available_nodes)) or 1
        layers_per_shard = max(1, num_layers // num_shards)
        assignments = {}
        for i in range(num_shards):
            node = available_nodes[i % len(available_nodes)]
            node_id = node.get("node_id", f"node-{i}")
            start = i * layers_per_shard
            end = min(start + layers_per_shard, num_layers)
            if i == num_shards - 1:
                end = num_layers
            layers = [f"layer_{j}" for j in range(start, end)]
            device = DeviceType.CPU
            if node.get("gpu_count", 0) > 0:
                device = DeviceType.CUDA
            sa = ShardAssignment(
                shard_id=f"shard-{job.job_id}-{i}",
                node_id=node_id,
                layers=layers,
                device=device,
            )
            assignments[sa.shard_id] = sa
        self.assignments[job.job_id] = assignments
        job.shard_assignments = assignments
        return assignments

    def reassign_failed_shard(self, job: TrainingJob, shard_id: str, new_node_id: str) -> ShardAssignment | None:
        if shard_id not in self.assignments.get(job.job_id, {}):
            return None
        sa = self.assignments[job.job_id][shard_id]
        sa.node_id = new_node_id
        sa.status = "reassigned"
        job.shard_assignments[shard_id] = sa
        return sa


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, job: TrainingJob, step: int, model_state: dict[str, np.ndarray], loss: float, metrics: dict[str, float] | None = None) -> ModelCheckpoint:
        ckpt = job.create_checkpoint(step, loss, metrics)
        ckpt_dir = os.path.join(self.checkpoint_dir, job.job_id)
        os.makedirs(ckpt_dir, exist_ok=True)
        filepath = os.path.join(ckpt_dir, f"checkpoint-{step}.npz")
        tmp_dir = os.path.join(ckpt_dir, ".tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_npz = os.path.join(tmp_dir, f"checkpoint-{step}.npz")
        try:
            np.savez_compressed(tmp_npz, **model_state)
            shutil.move(tmp_npz, filepath)
        except Exception as e:
            logger.error("Failed to save checkpoint NPZ for job %s step %d: %s", job.job_id, step, e)
            if os.path.exists(tmp_npz):
                os.remove(tmp_npz)
            raise
        meta_path = os.path.join(ckpt_dir, f"metadata-{step}.json")
        meta_tmp = os.path.join(tmp_dir, f"metadata-{step}.json")
        try:
            with open(meta_tmp, "w") as f:
                json.dump(ckpt.model_dump(), f, indent=2, default=str)
            shutil.move(meta_tmp, meta_path)
        except Exception as e:
            logger.error("Failed to save checkpoint metadata for job %s step %d: %s", job.job_id, step, e)
            if os.path.exists(meta_tmp):
                os.remove(meta_tmp)
            raise
        finally:
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass
        file_size = os.path.getsize(filepath) / (1024 * 1024) if os.path.exists(filepath) else 0
        ckpt.file_path = filepath
        ckpt.file_size_mb = file_size
        return ckpt

    def load_checkpoint(self, job_id: str, step: int | None = None) -> tuple[dict[str, np.ndarray], ModelCheckpoint] | None:
        ckpt_dir = os.path.join(self.checkpoint_dir, job_id)
        if not os.path.exists(ckpt_dir):
            return None
        if step is not None:
            filepath = os.path.join(ckpt_dir, f"checkpoint-{step}.npz")
            meta_path = os.path.join(ckpt_dir, f"metadata-{step}.json")
        else:
            npz_files = sorted(
                [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint-") and f.endswith(".npz")],
                key=lambda x: int(x.replace("checkpoint-", "").replace(".npz", "")),
            )
            if not npz_files:
                return None
            filepath = os.path.join(ckpt_dir, npz_files[-1])
            latest_step = int(npz_files[-1].replace("checkpoint-", "").replace(".npz", ""))
            meta_path = os.path.join(ckpt_dir, f"metadata-{latest_step}.json")
        if not os.path.exists(filepath):
            return None
        metadata = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    metadata = ModelCheckpoint(**json.load(f))
            except Exception as e:
                logger.warning("Failed to load checkpoint metadata for %s: %s", job_id, e)
        data = dict(np.load(filepath))
        return data, metadata

    def list_checkpoints(self, job_id: str) -> list[dict[str, Any]]:
        ckpt_dir = os.path.join(self.checkpoint_dir, job_id)
        if not os.path.exists(ckpt_dir):
            return []
        result = []
        for f in sorted(os.listdir(ckpt_dir)):
            if f.startswith("checkpoint-") and f.endswith(".npz"):
                step_str = f.replace("checkpoint-", "").replace(".npz", "")
                try:
                    step = int(step_str)
                except ValueError:
                    continue
                fp = os.path.join(ckpt_dir, f)
                result.append({
                    "step": step,
                    "file": f,
                    "size_mb": os.path.getsize(fp) / (1024 * 1024),
                })
        return result

    def prune_checkpoints(self, job_id: str, keep: int = 3):
        ckpt_dir = os.path.join(self.checkpoint_dir, job_id)
        if not os.path.exists(ckpt_dir):
            return
        files = sorted(
            [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint-") and f.endswith(".npz")],
            key=lambda x: int(x.replace("checkpoint-", "").replace(".npz", "")),
        )
        while len(files) > keep:
            f = files.pop(0)
            os.remove(os.path.join(ckpt_dir, f))
            step_str = f.replace("checkpoint-", "").replace(".npz", "")
            for meta_pattern in [f"metadata-{step_str}.json", f.replace(".npz", "-meta.json")]:
                meta_path = os.path.join(ckpt_dir, meta_pattern)
                if os.path.exists(meta_path):
                    os.remove(meta_path)


class LocalTrainer:
    def __init__(self, config: TrainingConfig, checkpoint_manager: CheckpointManager | None = None):
        self.config = config
        self.job = TrainingJob(config)
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.lr_scheduler = LearningRateScheduler(
            base_lr=config.learning_rate,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps,
        )
        self._step = 0
        self._running = False
        self._model_params: dict[str, np.ndarray] = {}
        self._optimizer_state: dict[str, Any] = {}
        self._rng = np.random.default_rng(config.seed)

    def _init_model(self):
        for i in range(self.config.num_layers):
            h = self.config.hidden_size
            inter = self.config.intermediate_size
            for name, shape in [
                (f"layer_{i}.attn.q_weight", (h, h)),
                (f"layer_{i}.attn.k_weight", (h, h)),
                (f"layer_{i}.attn.v_weight", (h, h)),
                (f"layer_{i}.attn.out_weight", (h, h)),
                (f"layer_{i}.ffn.up_weight", (h, inter)),
                (f"layer_{i}.ffn.down_weight", (inter, h)),
                (f"layer_{i}.ln1.weight", (h,)),
                (f"layer_{i}.ln1.bias", (h,)),
                (f"layer_{i}.ln2.weight", (h,)),
                (f"layer_{i}.ln2.bias", (h,)),
            ]:
                std = 0.02
                self._model_params[name] = self._rng.normal(0, std, shape).astype(np.float32)
        self._model_params["embed.weight"] = self._rng.normal(
            0, 0.02, (self.config.vocab_size, self.config.hidden_size)
        ).astype(np.float32)
        self._model_params["output.weight"] = self._rng.normal(
            0, 0.02, (self.config.vocab_size, self.config.hidden_size)
        ).astype(np.float32)
        self._model_params["output.bias"] = np.zeros(self.config.vocab_size, dtype=np.float32)
        self._model_params["final_ln.weight"] = np.ones(self.config.hidden_size, dtype=np.float32)
        self._model_params["final_ln.bias"] = np.zeros(self.config.hidden_size, dtype=np.float32)

    def _init_optimizer(self):
        for name, param in self._model_params.items():
            self._optimizer_state[name] = {
                "m": np.zeros_like(param),
                "v": np.zeros_like(param),
                "t": 0,
            }

    def _forward_step(self) -> float:
        total_norm = 0.0
        for name, param in self._model_params.items():
            if param.ndim < 2:
                continue
            param_norm = float(np.linalg.norm(param))
            total_norm += param_norm
        base_loss = 10.0 / (1.0 + self._step * 0.001)
        noise = float(self._rng.normal(0, 0.05))
        return max(0.01, base_loss + noise)

    def _compute_gradients(self) -> dict[str, np.ndarray]:
        grads = {}
        for name, param in self._model_params.items():
            grad = self._rng.normal(0, 0.01, param.shape).astype(np.float32)
            grad *= 1.0 / (1.0 + self._step * 0.0001)
            grads[name] = grad
        return grads

    def _apply_gradients(self, grads: dict[str, np.ndarray]):
        lr = self.lr_scheduler.get_lr(self._step)
        for name, grad in grads.items():
            if name not in self._model_params:
                continue
            param = self._model_params[name]
            state = self._optimizer_state.get(name, {"m": np.zeros_like(param), "v": np.zeros_like(param), "t": 0})
            state["t"] += 1
            state["m"] = 0.9 * state["m"] + 0.1 * grad
            state["v"] = 0.999 * state["v"] + 0.001 * grad**2
            m_hat = state["m"] / (1 - 0.9**state["t"])
            v_hat = state["v"] / (1 - 0.999**state["t"])
            self._model_params[name] = param - lr * (m_hat / (np.sqrt(v_hat) + 1e-8) + self.config.weight_decay * param)
            self._optimizer_state[name] = state

    async def train_step(self) -> TrainingMetrics:
        if not self._model_params:
            self._init_model()
            self._init_optimizer()
        loss = self._forward_step()
        grads = self._compute_gradients()
        self._apply_gradients(grads)
        self._step += 1
        grad_norm = float(np.sqrt(sum(float(np.sum(g**2)) for g in grads.values())))
        lr = self.lr_scheduler.get_lr(self._step)
        elapsed_s = time.time() - (self.job.started_at or time.time())
        metrics = TrainingMetrics(
            step=self._step,
            epoch=self._step * self.config.micro_batch_size * self.config.gradient_accumulation_steps / 100000,
            loss=loss,
            lr=lr,
            grad_norm=grad_norm,
            throughput_tok_s=self.config.batch_size * self.config.max_seq_length / max(0.001, elapsed_s),
            elapsed_s=elapsed_s,
        )
        self.job.record_metrics(metrics)
        if self._step % self.config.checkpoint_interval == 0:
            self.checkpoint_manager.save_checkpoint(self.job, self._step, self._model_params, loss)
        return metrics

    async def train(self, callback=None) -> TrainingJob:
        self.job.status = TrainingStatus.RUNNING
        self.job.started_at = time.time()
        self._running = True
        try:
            while self._step < self.config.total_steps and self._running:
                metrics = await self.train_step()
                if callback:
                    await callback(metrics)
                if self.job._stop_event.is_set():
                    self.job.status = TrainingStatus.CANCELLED
                    break
                await asyncio.sleep(0)
            if self.job.status == TrainingStatus.RUNNING:
                self.job.status = TrainingStatus.COMPLETED
                self.job.completed_at = time.time()
                self.checkpoint_manager.save_checkpoint(
                    self.job, self._step, self._model_params, self.job.best_loss
                )
        except Exception as e:
            self.job.status = TrainingStatus.FAILED
            self.job.error_message = str(e)
            logger.error("Training failed: %s", e)
        finally:
            self._running = False
        return self.job

    def stop(self):
        self._running = False
        self.job._stop_event.set()

    def get_model_state(self) -> dict[str, np.ndarray]:
        return copy.deepcopy(self._model_params)

    def load_model_state(self, state: dict[str, np.ndarray], preserve_optimizer: bool = False):
        self._model_params = copy.deepcopy(state)
        if not preserve_optimizer:
            self._init_optimizer()


class GradientSyncServer:
    def __init__(self, node_id: str, peer_endpoints: dict[str, str] | None = None):
        self.node_id = node_id
        self.peer_endpoints: dict[str, str] = peer_endpoints or {}
        self._gradient_store: dict[str, dict[int, dict[str, GradientShard]]] = {}
        self._sync_buffer: dict[str, dict[int, dict[str, np.ndarray]]] = {}
        self._aggregated_steps: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._sync_task: asyncio.Task | None = None
        self._sync_interval_s: float = 5.0
        self._compression_ratio: float = 0.01
        self._compression_method: str = "topk"

    def add_peer(self, node_id: str, endpoint: str):
        self.peer_endpoints[node_id] = endpoint

    def remove_peer(self, node_id: str):
        self.peer_endpoints.pop(node_id, None)

    async def start(self):
        self._running = True
        self._sync_task = asyncio.create_task(self._periodic_sync())

    async def stop(self):
        self._running = False
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

    async def push_gradients(self, job_id: str, step: int, gradients: dict[str, np.ndarray], node_id: str = ""):
        shard_data = {}
        sender = node_id or self.node_id
        for layer_name, grad in gradients.items():
            shard = GradientShard(
                shard_id=f"{sender}-{layer_name}-{step}",
                job_id=job_id,
                step=step,
                layer_name=layer_name,
                shape=list(grad.shape),
                data=grad.flatten().tolist(),
                num_samples=1,
                node_id=sender,
            )
            shard_data[layer_name] = shard
        async with self._lock:
            if job_id not in self._gradient_store:
                self._gradient_store[job_id] = {}
            if step not in self._gradient_store[job_id]:
                self._gradient_store[job_id][step] = {}
            self._gradient_store[job_id][step][sender] = shard_data

    async def pull_aggregated(self, job_id: str, step: int) -> dict[str, np.ndarray] | None:
        async with self._lock:
            if job_id not in self._sync_buffer:
                return None
            if step not in self._sync_buffer[job_id]:
                return None
            return {k: v.copy() for k, v in self._sync_buffer[job_id][step].items()}

    async def aggregate_for_step(self, job_id: str, step: int) -> dict[str, np.ndarray]:
        async with self._lock:
            step_shards = self._gradient_store.get(job_id, {}).get(step, {})
            if not step_shards:
                return {}
            result: dict[str, np.ndarray] = {}
            nodes_present = set()
            for node_id, shard_data in step_shards.items():
                nodes_present.add(node_id)
                for layer_name, shard in shard_data.items():
                    arr = np.array(shard.data, dtype=np.float32)
                    if shard.shape and len(shard.data) == 1:
                        arr = np.full(shard.shape, shard.data[0], dtype=np.float32)
                    elif shard.shape:
                        try:
                            arr = arr.reshape(shard.shape)
                        except ValueError:
                            if arr.size == 1 and len(shard.shape) > 0:
                                arr = np.full(shard.shape, arr.flat[0], dtype=np.float32)
                            else:
                                logger.warning("Cannot reshape shard %s: expected %s, got %d elements",
                                             shard.shard_id, shard.shape, arr.size)
                                continue
                    if layer_name in result:
                        if result[layer_name].shape == arr.shape:
                            result[layer_name] = result[layer_name] + arr
                        else:
                            logger.warning("Shape mismatch for %s: %s vs %s, using first node's shape",
                                          layer_name, result[layer_name].shape, arr.shape)
                    else:
                        result[layer_name] = arr.copy()
            num_nodes = max(len(nodes_present), 1)
            for layer_name in result:
                result[layer_name] = result[layer_name] / num_nodes
            if job_id not in self._sync_buffer:
                self._sync_buffer[job_id] = {}
            self._sync_buffer[job_id][step] = result
            self._aggregated_steps[job_id] = step
            return result

    async def send_gradients_to_peer(self, peer_id: str, job_id: str, step: int, gradients: dict[str, np.ndarray]) -> bool:
        if peer_id not in self.peer_endpoints:
            return False
        endpoint = self.peer_endpoints[peer_id]
        payload = {
            "job_id": job_id,
            "step": step,
            "node_id": self.node_id,
            "gradients": {},
        }
        for layer_name, grad in gradients.items():
            compressed = GradientCompressor.compress(grad, self._compression_method, self._compression_ratio)
            payload["gradients"][layer_name] = {
                "shape": list(grad.shape),
                "compressed": compressed,
                "hash": hashlib.sha256(grad.tobytes()).hexdigest()[:16],
            }
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{endpoint}/api/training/gradient-sync",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.error("Failed to send gradients to %s: %s", peer_id, e)
            return False

    async def receive_gradients(self, payload: dict[str, Any]) -> bool:
        job_id = payload.get("job_id", "")
        step = payload.get("step", 0)
        sender_id = payload.get("node_id", "")
        gradients_data = payload.get("gradients", {})
        gradients: dict[str, np.ndarray] = {}
        for layer_name, data in gradients_data.items():
            if isinstance(data, (list, np.ndarray)):
                gradients[layer_name] = np.array(data, dtype=np.float32)
            elif isinstance(data, dict):
                compressed = data.get("compressed", {})
                if compressed.get("method") not in ("none", ""):
                    decompressed = GradientCompressor.decompress(compressed)
                    gradients[layer_name] = decompressed
                else:
                    shape = data.get("shape", [])
                    values = compressed.get("values", data.get("values", []))
                    arr = np.array(values, dtype=np.float32)
                    if shape:
                        arr = arr.reshape(shape)
                    gradients[layer_name] = arr
        if gradients:
            await self.push_gradients(job_id, step, gradients, node_id=sender_id)
            return True
        return False

    async def _periodic_sync(self):
        while self._running:
            try:
                await asyncio.sleep(self._sync_interval_s)
                for job_id in list(self._gradient_store.keys()):
                    steps = self._gradient_store.get(job_id, {})
                    if steps:
                        latest_step = max(steps.keys(), default=-1)
                        if latest_step >= 0:
                            agg = self._aggregated_steps.get(job_id, -1)
                            if latest_step > agg:
                                await self.aggregate_for_step(job_id, latest_step)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Gradient sync error: %s", e)

    def get_sync_status(self) -> dict[str, Any]:
        status = {}
        for job_id, steps in self._gradient_store.items():
            status[job_id] = {
                "steps_available": sorted(steps.keys()),
                "nodes_per_step": {
                    step: list(shards.keys()) for step, shards in steps.items()
                },
                "aggregated_step": self._aggregated_steps.get(job_id, -1),
            }
        return {
            "node_id": self.node_id,
            "peers": len(self.peer_endpoints),
            "gradient_store": status,
            "running": self._running,
        }