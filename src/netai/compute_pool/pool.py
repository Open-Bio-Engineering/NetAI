"""ComputePool — the main orchestrator for distributed inference across volunteer nodes.

This is the central piece: nodes jack in, get assigned pipeline stages,
and participate in PPLNS share tracking. Anyone can run inference for free
 proportional to their contribution.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from netai.compute_pool.share import (
    ComputeContribution, ComputeShare, ShareDifficulty,
    ShareLedger, PPLNSRewardCalculator, ProofOfCompute, ShareStatus,
)
from netai.compute_pool.pipeline import (
    PipelineConfig, PipelineOrchestrator, PipelinePlan,
    PipelineStage, PipelineStatus, ActivationBuffer,
)
from netai.compute_pool.stratum import (
    StratumServer, StratumClient, StratumMessage,
    WorkAssignment, WorkResult, NodeDifficulty,
)

logger = logging.getLogger(__name__)


class PoolNodeStatus(str, Enum):
    CONNECTING = "connecting"
    SUBSCRIBING = "subscribing"
    AUTHORIZING = "authorizing"
    IDLE = "idle"
    WORKING = "working"
    DRAINING = "draining"
    OFFLINE = "offline"


class PoolNode(BaseModel):
    node_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    endpoint: str = ""
    status: PoolNodeStatus = PoolNodeStatus.OFFLINE
    gpu_count: int = 0
    gpu_names: list[str] = Field(default_factory=list)
    vram_total_mb: float = 0.0
    vram_available_mb: float = 0.0
    cpu_cores: int = 0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    compute_capability: str = ""
    joined_at: float = 0.0
    last_heartbeat: float = Field(default_factory=time.time)
    current_model: str = ""
    current_stage: int = -1
    inference_count: int = 0
    total_latency_ms: float = 0.0
    shares_submitted: int = 0
    difficulty: ShareDifficulty = ShareDifficulty.MEDIUM

    @property
    def is_available(self) -> bool:
        if self.status not in (PoolNodeStatus.IDLE, PoolNodeStatus.WORKING):
            return False
        if self.vram_available_mb <= 0:
            return False
        if self.last_heartbeat > 0 and (time.time() - self.last_heartbeat) > 300:
            return False
        return True

    @property
    def inference_capacity_score(self) -> float:
        score = self.cpu_cores * 0.5
        score += self.gpu_count * 10.0
        score += min(self.vram_available_mb / 1024.0, 64.0) * 2.0
        score += min(self.ram_available_gb, 128.0) * 0.2
        return score

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.inference_count, 1)


class PoolStatus(BaseModel):
    pool_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    total_nodes: int = 0
    active_nodes: int = 0
    total_vram_gb: float = 0.0
    available_vram_gb: float = 0.0
    total_gpus: int = 0
    models_loaded: int = 0
    pipelines_active: int = 0
    total_inferences: int = 0
    total_shares: int = 0
    uptime_seconds: float = 0.0
    started_at: float = Field(default_factory=time.time)


class ComputePool:
    def __init__(self, pool_id: str = ""):
        self.pool_id = pool_id or uuid.uuid4().hex[:8]
        self.nodes: dict[str, PoolNode] = {}
        self.orchestrator = PipelineOrchestrator(node_id=self.pool_id)
        self.ledger = ShareLedger()
        self.stratum = StratumServer(self.orchestrator, self.ledger)
        self.reward_calculator = PPLNSRewardCalculator()
        self._status = PoolStatus(pool_id=self.pool_id)
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

    async def start(self):
        self._running = True
        self._status.started_at = time.time()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Compute pool %s started", self.pool_id)

    async def stop(self):
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        logger.info("Compute pool %s stopped", self.pool_id)

    async def _heartbeat_loop(self):
        while self._running:
            try:
                for node_id, node in list(self.nodes.items()):
                    if node.status in (PoolNodeStatus.IDLE, PoolNodeStatus.WORKING):
                        self.orchestrator.heartbeat(
                            node.current_model, node.current_stage,
                            node.inference_count, 0,
                        )
                await asyncio.sleep(30.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat loop error: %s", e)

    async def _cleanup_loop(self):
        while self._running:
            try:
                now = time.time()
                stale_threshold = 300.0
                for node_id, node in list(self.nodes.items()):
                    if node.last_heartbeat > 0 and now - node.last_heartbeat > stale_threshold:
                        if node.status == PoolNodeStatus.WORKING:
                            node.status = PoolNodeStatus.DRAINING
                            logger.warning("Node %s stale, draining", node_id)
                        elif node.status in (PoolNodeStatus.IDLE, PoolNodeStatus.CONNECTING):
                            node.status = PoolNodeStatus.OFFLINE
                self.ledger.prune_expired(now)
                self._cleanup_old_activations()
                await asyncio.sleep(60.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup loop error: %s", e)

    def _cleanup_old_activations(self):
        for model_id in list(self.orchestrator.pipelines.keys()):
            plan = self.orchestrator.pipelines[model_id]
            if plan.status in (PipelineStatus.IDLE, PipelineStatus.ERROR):
                continue

    def register_node(self, node: PoolNode) -> PoolNode:
        node.joined_at = time.time()
        node.last_heartbeat = time.time()
        node.status = PoolNodeStatus.IDLE
        self.nodes[node.node_id] = node
        self._update_pool_status()
        logger.info("Node %s registered (GPUs=%d, VRAM=%.1fGB)", node.node_id, node.gpu_count, node.vram_total_mb / 1024)
        return node

    def unregister_node(self, node_id: str):
        node = self.nodes.pop(node_id, None)
        if node:
            logger.info("Node %s unregistered", node_id)
        self._update_pool_status()

    def update_heartbeat(self, node_id: str, vram_available_mb: float | None = None,
                         inference_count: int = 0, latency_ms: float = 0.0):
        node = self.nodes.get(node_id)
        if not node:
            return
        node.last_heartbeat = time.time()
        if vram_available_mb is not None:
            node.vram_available_mb = vram_available_mb
        if inference_count > 0:
            node.inference_count += inference_count
        if latency_ms > 0:
            node.total_latency_ms += latency_ms

    async def jack_in(self, node: PoolNode, model_id: str = "") -> dict[str, Any]:
        node = self.register_node(node)
        caps = {
            "vram_gb": node.vram_total_mb / 1024.0,
            "gpu_count": node.gpu_count,
            "cpu_cores": node.cpu_cores,
        }
        sub = await self.stratum.handle_subscribe(node.node_id, caps)
        try:
            node.difficulty = ShareDifficulty(sub.get("difficulty", "medium"))
        except ValueError:
            node.difficulty = ShareDifficulty.MEDIUM

        if model_id:
            stage = self.orchestrator.get_stage_for_node(model_id, node.node_id)
            if stage:
                node.current_model = model_id
                node.current_stage = stage.stage_index
                node.status = PoolNodeStatus.WORKING
                assignment = await self.stratum.assign_work(model_id, node.node_id)
                self._update_pool_status()
                return {
                    "status": "assigned",
                    "subscription": sub,
                    "stage": stage.model_dump(),
                    "assignment": assignment.model_dump() if assignment else None,
                }

        self._update_pool_status()
        return {"status": "subscribed", "subscription": sub}

    async def jack_out(self, node_id: str) -> bool:
        node = self.nodes.get(node_id)
        if not node:
            return False
        if node.current_model:
            await self.orchestrator.remove_node_from_pipeline(node.current_model, node_id)
            plan = self.orchestrator.get_pipeline(node.current_model)
            if plan and all(s.status.value == "idle" and not s.node_id for s in plan.stages):
                await self.orchestrator.remove_pipeline(node.current_model)
        node.status = PoolNodeStatus.DRAINING
        self.unregister_node(node_id)
        return True

    def plan_model(self, config: PipelineConfig) -> PipelinePlan:
        node_resources = [
            {
                "node_id": n.node_id,
                "vram_available_mb": n.vram_available_mb,
                "gpu_count": n.gpu_count,
                "cpu_cores": n.cpu_cores,
            }
            for n in self.nodes.values()
            if n.is_available
        ]
        plan = self.orchestrator.plan_pipeline(config, node_resources)
        self._update_pool_status()
        return plan

    def get_inference_credits(self, node_id: str) -> float:
        rewards = self.reward_calculator.calculate_rewards(self.ledger, total_inference_budget=1_000_000)
        return rewards.get(node_id, 0.0)

    def get_contribution(self, node_id: str) -> ComputeContribution | None:
        return self.ledger.get_contribution(node_id)

    def get_leaderboard(self, limit: int = 50) -> list[ComputeContribution]:
        return self.ledger.get_leaderboard(limit)

    def can_run_model(self, model_size_mb: float) -> tuple[bool, dict[str, Any]]:
        available_nodes = [n for n in self.nodes.values() if n.is_available]
        available_vram = sum(n.vram_available_mb for n in available_nodes)
        available_count = len(available_nodes)
        total_vram_with_pipeline = available_vram * 0.85
        can = total_vram_with_pipeline >= model_size_mb or available_count >= 2
        return can, {
            "available_vram_mb": available_vram,
            "available_nodes": available_count,
            "model_size_mb": model_size_mb,
            "pipeline_needed": available_vram < model_size_mb,
        }

    def _update_pool_status(self):
        self._status.total_nodes = len(self.nodes)
        self._status.active_nodes = sum(1 for n in self.nodes.values() if n.is_available)
        self._status.total_vram_gb = sum(n.vram_total_mb for n in self.nodes.values()) / 1024
        self._status.available_vram_gb = sum(n.vram_available_mb for n in self.nodes.values() if n.is_available) / 1024
        self._status.total_gpus = sum(n.gpu_count for n in self.nodes.values())
        self._status.models_loaded = len(self.orchestrator.pipelines)
        self._status.pipelines_active = sum(1 for p in self.orchestrator.pipelines.values() if p.status == PipelineStatus.RUNNING)
        self._status.total_inferences = sum(n.inference_count for n in self.nodes.values())
        self._status.total_shares = self.ledger.total_share_count()
        self._status.uptime_seconds = time.time() - self._status.started_at

    def get_status(self) -> PoolStatus:
        self._update_pool_status()
        return self._status