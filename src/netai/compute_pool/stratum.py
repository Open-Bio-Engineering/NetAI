"""Stratum-like protocol for distributing inference work across volunteer nodes.

Adapted from the Stratum mining protocol (used by all mining pools since 2012)
for AI compute: instead of hashing work, we distribute tensor-compute work.
Each connected node gets assigned pipeline stages matching its capability,
receives activation tensors, processes them, and returns results.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from netai.compute_pool.share import (
    ComputeShare, ShareDifficulty, ShareLedger, ProofOfCompute, ShareStatus,
)
from netai.compute_pool.pipeline import (
    PipelineConfig, PipelineOrchestrator, PipelineStage, PipelineStatus,
)

logger = logging.getLogger(__name__)


class StratumMessageType(str, Enum):
    SUBSCRIBE = "mining.subscribe"
    AUTHORIZE = "mining.authorize"
    SUBMIT = "mining.submit"
    NOTIFY = "mining.notify"
    SET_DIFFICULTY = "mining.set_difficulty"
    JOB_ASSIGN = "job.assign"
    JOB_RESULT = "job.result"
    HEARTBEAT = "heartbeat"
    CONFIGURE = "configure"
    PING = "ping"
    PONG = "pong"


class WorkAssignment(BaseModel):
    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_id: str = ""
    stage_index: int = 0
    layer_start: int = 0
    layer_end: int = 0
    input_hash: str = ""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    created_at: float = Field(default_factory=time.time)
    expires_at: float = 0.0

    @property
    def is_expired(self) -> bool:
        return self.expires_at > 0 and time.time() > self.expires_at


class WorkResult(BaseModel):
    job_id: str = ""
    node_id: str = ""
    model_id: str = ""
    stage_index: int = 0
    output_hash: str = ""
    output_tokens: int = 0
    latency_ms: float = 0.0
    vram_used_mb: float = 0.0
    compute_steps: int = 0
    proof: ProofOfCompute | None = None
    timestamp: float = Field(default_factory=time.time)
    error: str | None = None


class NodeDifficulty(BaseModel):
    node_id: str = ""
    current_difficulty: ShareDifficulty = ShareDifficulty.MEDIUM
    target_shares_per_minute: float = 1.0
    vram_gb: float = 0.0
    gpu_count: int = 0
    cpu_cores: int = 0
    last_adjusted: float = Field(default_factory=time.time)
    _share_history: list[tuple[float, float]] = []

    model_config = {"arbitrary_types_allowed": True}

    def adjust(self, shares_last_minute: float):
        self._share_history.append((time.time(), shares_last_minute))
        if len(self._share_history) > 60:
            self._share_history = self._share_history[-60:]
        recent = self._share_history[-min(len(self._share_history), 5):]
        avg = sum(s for _, s in recent) / len(recent)
        if avg > self.target_shares_per_minute * 2:
            if self.current_difficulty == ShareDifficulty.LIGHT:
                self.current_difficulty = ShareDifficulty.MEDIUM
            elif self.current_difficulty == ShareDifficulty.MEDIUM:
                self.current_difficulty = ShareDifficulty.HEAVY
        elif avg < self.target_shares_per_minute * 0.5:
            if self.current_difficulty == ShareDifficulty.HEAVY:
                self.current_difficulty = ShareDifficulty.MEDIUM
            elif self.current_difficulty == ShareDifficulty.MEDIUM:
                self.current_difficulty = ShareDifficulty.LIGHT
        self.last_adjusted = time.time()


class StratumMessage(BaseModel):
    id: int = 0
    method: str = ""
    params: list[Any] = Field(default_factory=list)
    result: Any = None
    error: list | None = None

    @staticmethod
    def subscribe(node_id: str, capabilities: dict[str, Any] | None = None) -> "StratumMessage":
        return StratumMessage(
            id=1,
            method=StratumMessageType.SUBSCRIBE,
            params=[node_id, capabilities or {}],
        )

    @staticmethod
    def authorize(user_id: str, token: str) -> "StratumMessage":
        return StratumMessage(
            id=2,
            method=StratumMessageType.AUTHORIZE,
            params=[user_id, token],
        )

    @staticmethod
    def submit(share: ComputeShare) -> "StratumMessage":
        return StratumMessage(
            id=3,
            method=StratumMessageType.SUBMIT,
            params=[share.node_id, share.model_id, share.stage_index, share.difficulty.value],
        )

    @staticmethod
    def notify(job_id: str, assignment: WorkAssignment) -> "StratumMessage":
        return StratumMessage(
            id=0,
            method=StratumMessageType.NOTIFY,
            params=[job_id, assignment.model_id, assignment.stage_index, assignment.input_hash],
        )

    @staticmethod
    def set_difficulty(difficulty: ShareDifficulty) -> "StratumMessage":
        return StratumMessage(
            id=0,
            method=StratumMessageType.SET_DIFFICULTY,
            params=[difficulty.value],
        )


class StratumServer:
    def __init__(self, orchestrator: PipelineOrchestrator, ledger: ShareLedger):
        self.orchestrator = orchestrator
        self.ledger = ledger
        self._connections: dict[str, dict[str, Any]] = {}
        self._difficulties: dict[str, NodeDifficulty] = {}
        self._job_counter = 0
        self._running = False

    async def handle_subscribe(self, node_id: str, capabilities: dict[str, Any]) -> dict[str, Any]:
        vram_gb = capabilities.get("vram_gb", 0)
        difficulty = ShareDifficulty.from_vram_gb(vram_gb)
        node_diff = NodeDifficulty(
            node_id=node_id,
            current_difficulty=difficulty,
            vram_gb=vram_gb,
            gpu_count=capabilities.get("gpu_count", 0),
            cpu_cores=capabilities.get("cpu_cores", 0),
        )
        self._difficulties[node_id] = node_diff
        self._connections[node_id] = {
            "capabilities": capabilities,
            "subscribed_at": time.time(),
            "last_seen": time.time(),
        }
        subscription_id = uuid.uuid4().hex[:16]
        logger.info("Node %s subscribed (difficulty=%s, vram=%.1fGB)", node_id, difficulty.value, vram_gb)
        return {
            "subscription_id": subscription_id,
            "difficulty": difficulty.value,
            "node_id": node_id,
        }

    async def handle_authorize(self, user_id: str, token: str) -> dict[str, Any]:
        return {
            "authorized": True,
            "user_id": user_id,
            "message": "Authorization accepted",
        }

    async def handle_submit(self, result: WorkResult) -> ComputeShare | None:
        if result.proof is None:
            logger.warning("Submit from %s missing proof", result.node_id)
            return None
        node_diff = self._difficulties.get(result.node_id)
        difficulty = node_diff.current_difficulty if node_diff else ShareDifficulty.MEDIUM
        share = ComputeShare(
            node_id=result.node_id,
            model_id=result.model_id,
            stage_index=result.stage_index,
            difficulty=difficulty,
            proof=result.proof,
            latency_ms=result.latency_ms,
        )
        if not self.ledger.validate_share(share, difficulty_target=difficulty.difficulty_target()):
            share.status = ShareStatus.INVALID
            logger.warning("Invalid share from %s", result.node_id)
            return share
        share = self.ledger.add_share(share)
        logger.info("Valid share from %s (weight=%.2f, tokens=%.0f)", result.node_id, share.weight, share.tokens_contributed)
        if node_diff:
            recent = len([s for s in self.ledger.get_recent_shares(60, result.node_id) if s.node_id == result.node_id])
            node_diff.adjust(recent)
        return share

    async def assign_work(self, model_id: str, node_id: str) -> WorkAssignment | None:
        stage = self.orchestrator.get_stage_for_node(model_id, node_id)
        if not stage:
            return None
        self._job_counter += 1
        assignment = WorkAssignment(
            job_id=f"job_{self._job_counter:06d}",
            model_id=model_id,
            stage_index=stage.stage_index,
            layer_start=stage.layer_start,
            layer_end=stage.layer_end,
            expires_at=time.time() + 300,
        )
        return assignment

    def get_status(self) -> dict[str, Any]:
        return {
            "connections": len(self._connections),
            "difficulty_overview": {
                nid: {"difficulty": nd.current_difficulty.value, "vram_gb": nd.vram_gb}
                for nid, nd in self._difficulties.items()
            },
            "total_shares": self.ledger.total_share_count(),
        }


class StratumClient:
    def __init__(self, server_url: str, node_id: str, capabilities: dict[str, Any] | None = None):
        self.server_url = server_url
        self.node_id = node_id
        self.capabilities = capabilities or {}
        self._connected = False
        self._subscription_id: str | None = None
        self._current_difficulty: ShareDifficulty = ShareDifficulty.MEDIUM
        self._current_job: WorkAssignment | None = None
        self._share_count = 0

    async def connect_and_subscribe(self) -> dict[str, Any]:
        self._connected = True
        subscription = {
            "subscription_id": uuid.uuid4().hex[:16],
            "difficulty": self._current_difficulty.value,
            "node_id": self.node_id,
        }
        self._subscription_id = subscription["subscription_id"]
        return subscription

    async def authorize(self, user_id: str, token: str) -> bool:
        return True

    async def submit_result(self, result: WorkResult) -> ComputeShare | None:
        self._share_count += 1
        return ComputeShare(
            node_id=self.node_id,
            model_id=result.model_id,
            stage_index=result.stage_index,
            difficulty=self._current_difficulty,
            latency_ms=result.latency_ms,
        )

    async def receive_work(self) -> WorkAssignment | None:
        return self._current_job

    def update_difficulty(self, new_difficulty: ShareDifficulty):
        self._current_difficulty = new_difficulty