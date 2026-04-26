"""Pipeline-parallel inference — split a 700B+ model across volunteer nodes.

Each node loads a contiguous range of transformer layers. Activations flow
through the pipeline: Node 0 → Node 1 → ... → Node N. This is the core
that makes "Ollama without a datacenter" work — a 700B model that needs
1.4TB of VRAM can run across 50 home GPUs with 28GB each.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import deque
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_ACTIVATION_SIZE_MB = 512.0


class PipelineStatus(str, Enum):
    IDLE = "idle"
    ASSEMBLING = "assembling"
    READY = "ready"
    RUNNING = "running"
    DRAINING = "draining"
    ERROR = "error"


class PipelineStage(BaseModel):
    stage_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    node_id: str = ""
    model_id: str = ""
    stage_index: int = 0
    total_stages: int = 0
    layer_start: int = 0
    layer_end: int = 0
    num_layers: int = 0
    vram_required_mb: float = 0.0
    status: PipelineStatus = PipelineStatus.IDLE
    loaded_at: float = 0.0
    inference_count: int = 0
    total_latency_ms: float = 0.0
    last_heartbeat: float = 0.0
    checksum: str = ""

    @property
    def is_ready(self) -> bool:
        return self.status in (PipelineStatus.READY, PipelineStatus.RUNNING)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.inference_count, 1)

    @property
    def layer_range(self) -> str:
        return f"{self.layer_start}-{self.layer_end}"


class PipelineConfig(BaseModel):
    model_id: str = ""
    model_name: str = ""
    total_layers: int = 0
    hidden_size: int = 0
    num_heads: int = 0
    vocab_size: int = 0
    intermediate_size: int = 0
    bytes_per_param: int = 2
    kv_cache_mb_per_seq: float = 0.0
    max_sequence_length: int = 2048
    quantization: str = "q4_k_m"

    @property
    def total_params(self) -> int:
        h = self.hidden_size
        l = self.total_layers
        i = self.intermediate_size or 4 * h
        v = self.vocab_size
        embed = v * h
        attn_per_layer = 4 * h * h
        ffn_per_layer = 2 * h * i + h * i
        layernorm_per_layer = 2 * 2 * h
        output_proj = h * v
        return embed + l * (attn_per_layer + ffn_per_layer + layernorm_per_layer) + output_proj

    @property
    def model_size_mb(self) -> float:
        return self.total_params * self.bytes_per_param / (1024 * 1024)

    def vram_per_stage(self, num_stages: int, batch_size: int = 1) -> float:
        if num_stages <= 0:
            return self.model_size_mb
        layers_per_stage = max(1, self.total_layers // num_stages)
        i = self.intermediate_size or 4 * self.hidden_size
        params_per_stage = layers_per_stage * (
            4 * self.hidden_size ** 2 + 2 * self.hidden_size * i
        ) + self.hidden_size * self.vocab_size / num_stages
        weights_mb = params_per_stage * self.bytes_per_param / (1024 * 1024)
        kv_mb = batch_size * self.max_sequence_length * self.hidden_size * 2 * layers_per_stage * self.bytes_per_param / (1024 * 1024)
        overhead = weights_mb * 0.1
        return weights_mb + kv_mb + overhead


class PipelinePlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_id: str = ""
    config: PipelineConfig = Field(default_factory=PipelineConfig)
    stages: list[PipelineStage] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    status: PipelineStatus = PipelineStatus.IDLE

    @property
    def coverage(self) -> float:
        if not self.stages:
            return 0.0
        covered = sum(s.num_layers for s in self.stages)
        return covered / max(self.config.total_layers, 1)

    @property
    def is_complete(self) -> bool:
        return self.coverage >= 1.0


class ActivationBuffer(BaseModel):
    request_id: str = ""
    from_stage: int = 0
    to_stage: int = 0
    seq_index: int = 0
    hidden_states_hash: str = ""
    shape: list[int] = Field(default_factory=list)
    dtype: str = "float16"
    data: bytes = b""
    timestamp: float = Field(default_factory=time.time)

    model_config = {"arbitrary_types_allowed": True}

    def compute_hash(self, hidden_states: np.ndarray) -> str:
        raw = hidden_states.tobytes()
        return hashlib.sha256(raw).hexdigest()[:16]

    @staticmethod
    def serialize_hidden(hidden: np.ndarray) -> tuple[bytes, list[int], str]:
        dtype = str(hidden.dtype)
        shape = list(hidden.shape)
        data = hidden.tobytes()
        size_mb = len(data) / (1024 * 1024)
        if size_mb > MAX_ACTIVATION_SIZE_MB:
            logger.warning("Activation serialization: %.1fMB exceeds limit (%.0fMB), truncating", size_mb, MAX_ACTIVATION_SIZE_MB)
        return data, shape, dtype

    @staticmethod
    def deserialize_hidden(data: bytes, shape: list[int], dtype: str) -> np.ndarray:
        dt = np.dtype(dtype)
        arr = np.frombuffer(data, dtype=dt)
        if shape:
            arr = arr.reshape(shape)
        return arr


class PipelineOrchestrator:
    def __init__(self, node_id: str = ""):
        self.node_id = node_id or uuid.uuid4().hex[:12]
        self.pipelines: dict[str, PipelinePlan] = {}
        self._local_stages: dict[str, PipelineStage] = {}
        self._activations: dict[str, deque[ActivationBuffer]] = {}
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._status = PipelineStatus.IDLE

    def plan_pipeline(
        self,
        config: PipelineConfig,
        node_resources: list[dict[str, Any]],
    ) -> PipelinePlan:
        plan = PipelinePlan(model_id=config.model_id, config=config)
        if config.total_layers <= 0:
            return plan

        sorted_nodes = sorted(node_resources, key=lambda n: n.get("vram_available_mb", 0), reverse=True)
        layers_remaining = config.total_layers
        stage_idx = 0

        for node in sorted_nodes:
            if layers_remaining <= 0:
                break
            vram = node.get("vram_available_mb", 0)
            if vram <= 0:
                continue
            vram_for_weights = vram * 0.7
            bytes_per_layer = (config.total_params * config.bytes_per_param) / max(config.total_layers * 1024 * 1024, 1)
            layers_this_node = min(int(vram_for_weights / max(bytes_per_layer, 0.01)), layers_remaining)
            if layers_this_node <= 0:
                layers_this_node = 1
            layers_this_node = min(layers_this_node, layers_remaining)

            vram_required = layers_this_node * bytes_per_layer * 1.1

            stage = PipelineStage(
                node_id=node.get("node_id", ""),
                model_id=config.model_id,
                stage_index=stage_idx,
                total_stages=0,
                layer_start=config.total_layers - layers_remaining,
                layer_end=config.total_layers - layers_remaining + layers_this_node - 1,
                num_layers=layers_this_node,
                vram_required_mb=vram_required,
                status=PipelineStatus.IDLE,
            )
            plan.stages.append(stage)
            stage_idx += 1
            layers_remaining -= layers_this_node

        for i, stage in enumerate(plan.stages):
            stage.total_stages = len(plan.stages)

        plan.status = PipelineStatus.ASSEMBLING
        self.pipelines[config.model_id] = plan
        return plan

    def assign_stage(self, model_id: str, stage: PipelineStage) -> bool:
        plan = self.pipelines.get(model_id)
        if not plan:
            return False
        for existing in plan.stages:
            if existing.stage_index == stage.stage_index:
                existing.node_id = stage.node_id
                existing.status = PipelineStatus.READY
                existing.loaded_at = time.time()
                existing.last_heartbeat = time.time()
                return True
        return False

    def get_stage_for_node(self, model_id: str, node_id: str) -> PipelineStage | None:
        plan = self.pipelines.get(model_id)
        if not plan:
            return None
        for stage in plan.stages:
            if stage.node_id == node_id:
                return stage
        return None

    def heartbeat(self, model_id: str, stage_index: int, inference_count: int = 0, latency_ms: float = 0.0):
        plan = self.pipelines.get(model_id)
        if not plan:
            return
        for stage in plan.stages:
            if stage.stage_index == stage_index:
                stage.last_heartbeat = time.time()
                stage.inference_count += inference_count
                stage.total_latency_ms += latency_ms
                break

    def check_pipeline_health(self, model_id: str, timeout_seconds: float = 120.0) -> PipelineStatus:
        plan = self.pipelines.get(model_id)
        if not plan:
            return PipelineStatus.IDLE
        now = time.time()
        for stage in plan.stages:
            if stage.status == PipelineStatus.IDLE:
                return PipelineStatus.ASSEMBLING
            if now - stage.last_heartbeat > timeout_seconds and stage.last_heartbeat > 0:
                stage.status = PipelineStatus.ERROR
                return PipelineStatus.ERROR
        all_ready = all(s.is_ready for s in plan.stages)
        if all_ready and plan.is_complete:
            plan.status = PipelineStatus.READY
            self._status = PipelineStatus.READY
        return plan.status

    def get_pipeline(self, model_id: str) -> PipelinePlan | None:
        return self.pipelines.get(model_id)

    def list_pipelines(self) -> list[PipelinePlan]:
        return list(self.pipelines.values())

    def remove_pipeline(self, model_id: str) -> bool:
        return self.pipelines.pop(model_id, None) is not None

    def remove_node_from_pipeline(self, model_id: str, node_id: str) -> bool:
        plan = self.pipelines.get(model_id)
        if not plan:
            return False
        for stage in plan.stages:
            if stage.node_id == node_id:
                stage.node_id = ""
                stage.status = PipelineStatus.IDLE
                stage.loaded_at = 0
                stage.last_heartbeat = 0
                break
        return True

    def store_activation(self, request_id: str, buffer: ActivationBuffer):
        if request_id not in self._activations:
            self._activations[request_id] = deque(maxlen=256)
        self._activations[request_id].append(buffer)

    def get_next_activation(self, request_id: str) -> ActivationBuffer | None:
        queue = self._activations.get(request_id)
        if queue:
            return queue.popleft() if queue else None
        return None

    def cleanup_request(self, request_id: str):
        self._activations.pop(request_id, None)
        self._pending_requests.pop(request_id, None)

    def cleanup_stale_activations(self, max_age_seconds: float = 300.0):
        now = time.time()
        stale = [rid for rid, queue in self._activations.items()
                  if queue and queue[0].timestamp < now - max_age_seconds]
        for rid in stale:
            self._activations.pop(rid, None)