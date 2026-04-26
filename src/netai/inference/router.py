"""Inference load balancer and request router for P2P distributed inference."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections import defaultdict
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from netai.inference.engine import (
    InferenceEngine, InferenceRequest, InferenceResponse,
    ModelServeConfig, ModelReplica, InferenceStatus, ModelMirror,
)

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    LOWEST_LATENCY = "lowest_latency"
    RANDOM = "random"
    HASH_BASED = "hash_based"
    ADAPTIVE = "adaptive"


class InferenceNode(BaseModel):
    node_id: str
    endpoint: str = ""
    status: InferenceStatus = InferenceStatus.IDLE
    models_loaded: list[str] = Field(default_factory=list)
    gpu_count: int = 0
    gpu_available: int = 0
    cpu_cores: int = 0
    ram_gb: float = 0.0
    capacity: int = 100
    current_load: int = 0
    avg_latency_ms: float = 0.0
    total_inferences: int = 0
    health_score: float = 1.0
    last_heartbeat: float = Field(default_factory=time.time)
    region: str = ""
    group_id: str = ""

    @property
    def is_available(self) -> bool:
        if self.status not in (InferenceStatus.READY, InferenceStatus.SERVING):
            return False
        if self.current_load >= self.capacity:
            return False
        if self.health_score <= 0.3:
            return False
        if self.last_heartbeat <= 0:
            return False
        if (time.time() - self.last_heartbeat) > 120:
            return False
        return True

    @property
    def load_factor(self) -> float:
        return self.current_load / max(self.capacity, 1)


class BatchedRequest(BaseModel):
    batch_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    requests: list[InferenceRequest] = Field(default_factory=list)
    model_id: str = ""
    created_at: float = Field(default_factory=time.time)


class InferenceLoadBalancer:
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.nodes: dict[str, InferenceNode] = {}
        self.model_nodes: dict[str, list[str]] = {}
        self.mirror: ModelMirror = ModelMirror()
        self._rr_counters: dict[str, int] = defaultdict(int)
        self._latency_history: dict[str, list[float]] = defaultdict(list)
        self._request_queue: asyncio.Queue | None = None
        self._batch_queue: asyncio.Queue | None = None
        self._batch_size = 8
        self._batch_timeout_ms = 50
        self._running = False
        self._session: Any = None
        self._session_lock = asyncio.Lock()

    async def start(self):
        self._request_queue = asyncio.Queue(maxsize=1000)
        self._batch_queue = asyncio.Queue(maxsize=100)
        self._running = True
        try:
            import aiohttp
            async with self._session_lock:
                if self._session is None:
                    self._session = aiohttp.ClientSession()
        except Exception:
            self._session = None

    async def stop(self):
        self._running = False
        async with self._session_lock:
            if self._session:
                await self._session.close()
                self._session = None

    def register_node(self, node: InferenceNode):
        self.nodes[node.node_id] = node
        for model_id in node.models_loaded:
            if model_id not in self.model_nodes:
                self.model_nodes[model_id] = []
            if node.node_id not in self.model_nodes[model_id]:
                self.model_nodes[model_id].append(node.node_id)
        logger.info("Inference node registered: %s (models=%s)", node.node_id, node.models_loaded)

    def unregister_node(self, node_id: str):
        node = self.nodes.pop(node_id, None)
        if node:
            for model_id in node.models_loaded:
                if model_id in self.model_nodes:
                    self.model_nodes[model_id] = [n for n in self.model_nodes[model_id] if n != node_id]
                    self.mirror.unregister_mirror(model_id, node_id)

    def update_node_heartbeat(self, node_id: str, load: int = 0, avg_latency: float = 0.0):
        node = self.nodes.get(node_id)
        if node:
            node.last_heartbeat = time.time()
            node.current_load = load
            if avg_latency > 0:
                node.avg_latency_ms = avg_latency
                self._latency_history[node_id].append(avg_latency)
                if len(self._latency_history[node_id]) > 100:
                    self._latency_history[node_id] = self._latency_history[node_id][-100:]
            if node.health_score < 1.0 and node.health_score > 0:
                node.health_score = min(1.0, node.health_score + 0.05)

    def add_model_to_node(self, node_id: str, model_id: str):
        node = self.nodes.get(node_id)
        if node and model_id not in node.models_loaded:
            node.models_loaded.append(model_id)
            self.model_nodes.setdefault(model_id, []).append(node_id)
            self.mirror.register_mirror(model_id, node_id)

    def route_request(self, request: InferenceRequest) -> str | None:
        model_id = request.model_id
        candidates = self._get_candidates(model_id, request.group_id)
        if not candidates:
            return None
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._route_round_robin(model_id, candidates)
        elif self.strategy == RoutingStrategy.LEAST_LOADED:
            return self._route_least_loaded(candidates)
        elif self.strategy == RoutingStrategy.LOWEST_LATENCY:
            return self._route_lowest_latency(candidates)
        elif self.strategy == RoutingStrategy.HASH_BASED:
            return self._route_hash_based(request, candidates)
        elif self.strategy == RoutingStrategy.RANDOM:
            return self._route_random(candidates)
        elif self.strategy == RoutingStrategy.ADAPTIVE:
            return self._route_adaptive(request, candidates)
        return candidates[0] if candidates else None

    def _get_candidates(self, model_id: str, group_id: str = "") -> list[str]:
        node_ids = self.model_nodes.get(model_id, [])
        candidates = []
        for nid in node_ids:
            node = self.nodes.get(nid)
            if node and node.is_available:
                if group_id and node.group_id and node.group_id != group_id:
                    continue
                candidates.append(nid)
        return candidates

    def _route_round_robin(self, model_id: str, candidates: list[str]) -> str:
        idx = self._rr_counters[model_id] % len(candidates)
        self._rr_counters[model_id] += 1
        return candidates[idx]

    def _route_least_loaded(self, candidates: list[str]) -> str:
        return min(candidates, key=lambda n: self.nodes[n].load_factor)

    def _route_lowest_latency(self, candidates: list[str]) -> str:
        return min(candidates, key=lambda n: self.nodes[n].avg_latency_ms)

    def _route_hash_based(self, request: InferenceRequest, candidates: list[str]) -> str:
        h = int(hashlib.md5(f"{request.user_id}:{request.model_id}".encode()).hexdigest(), 16)
        return candidates[h % len(candidates)]

    def _route_random(self, candidates: list[str]) -> str:
        return candidates[int(np.random.randint(0, len(candidates)))]

    def _route_adaptive(self, request: InferenceRequest, candidates: list[str]) -> str:
        scored = []
        for nid in candidates:
            node = self.nodes[nid]
            score = 100.0
            score -= node.load_factor * 30.0
            score -= node.avg_latency_ms / 100.0
            score += node.health_score * 20.0
            if node.gpu_count > 0:
                score += 10.0
            lat_history = self._latency_history.get(nid, [])
            if len(lat_history) > 5:
                recent_avg = np.mean(lat_history[-5:])
                score -= recent_avg / 50.0
            scored.append((nid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    async def submit_request(self, request: InferenceRequest, max_retries: int = 2) -> InferenceResponse:
        node_id = self.route_request(request)
        if not node_id:
            node_id = await self._try_mirror_or_queue(request)
            if not node_id:
                return InferenceResponse(
                    request_id=request.request_id,
                    error=f"No available nodes for model {request.model_id}",
                )
        last_error: str | None = None
        excluded: set[str] = set()
        for attempt in range(max_retries + 1):
            node = self.nodes.get(node_id)
            if not node:
                return InferenceResponse(request_id=request.request_id, error="Node not found")
            node.current_load += 1
            node.total_inferences += 1
            response = await self._forward_to_node(node, request)
            node.current_load = max(0, node.current_load - 1)
            if response.error is None:
                return response
            last_error = response.error
            excluded.add(node_id)
            fallback = self._route_fallback(request, excluded)
            if fallback is None or attempt >= max_retries:
                break
            node_id = fallback
        return InferenceResponse(request_id=request.request_id, error=last_error or "All retries exhausted")

    def _route_fallback(self, request: InferenceRequest, excluded: set[str]) -> str | None:
        candidates = self._get_candidates(request.model_id, request.group_id)
        candidates = [c for c in candidates if c not in excluded]
        if not candidates:
            return None
        return self._route_least_loaded(candidates)

    async def _try_mirror_or_queue(self, request: InferenceRequest) -> str | None:
        mirrors = self.mirror.get_mirrors(request.model_id)
        for mid in mirrors:
            if mid in self.nodes and self.nodes[mid].is_available:
                return mid
        return None

    async def _forward_to_node(self, node: InferenceNode, request: InferenceRequest) -> InferenceResponse:
        try:
            import aiohttp
            async with self._session_lock:
                if self._session is None:
                    self._session = aiohttp.ClientSession()
                session = self._session
            async with session.post(
                f"{node.endpoint}/api/inference/run",
                json=request.model_dump(),
                timeout=aiohttp.ClientTimeout(total=request.timeout_ms / 1000),
            ) as resp:
                if resp.status == 200:
                    return InferenceResponse(**await resp.json())
                return InferenceResponse(request_id=request.request_id, error=f"Node returned {resp.status}")
        except Exception as e:
            node.health_score = max(0.0, node.health_score * 0.9)
            return InferenceResponse(request_id=request.request_id, error=str(e))

    def get_status(self) -> dict[str, Any]:
        available = sum(1 for n in self.nodes.values() if n.is_available)
        total_inferences = sum(n.total_inferences for n in self.nodes.values())
        model_status = {}
        for mid, node_ids in self.model_nodes.items():
            avail = [nid for nid in node_ids if nid in self.nodes and self.nodes[nid].is_available]
            model_status[mid] = {
                "nodes": len(node_ids),
                "available": len(avail),
                "mirrors": self.mirror.get_mirror_count(mid),
            }
        return {
            "strategy": self.strategy.value,
            "total_nodes": len(self.nodes),
            "available_nodes": available,
            "total_inferences": total_inferences,
            "models": model_status,
            "mirroring": self.mirror.get_status(),
            "nodes": [
                {
                    "node_id": n.node_id,
                    "status": n.status.value,
                    "models": n.models_loaded,
                    "load": f"{n.current_load}/{n.capacity}",
                    "avg_latency_ms": round(n.avg_latency_ms, 2),
                    "health": round(n.health_score, 2),
                }
                for n in self.nodes.values()
            ],
        }


class InferenceGateway:
    def __init__(self, local_engine: InferenceEngine, load_balancer: InferenceLoadBalancer):
        self.local = local_engine
        self.lb = load_balancer
        self._running = False

    async def start(self):
        await self.local.start()
        await self.lb.start()
        self._running = True

    async def stop(self):
        await self.local.stop()
        await self.lb.stop()
        self._running = False

    async def serve(self, request: InferenceRequest) -> InferenceResponse:
        if not self._running:
            return InferenceResponse(request_id=request.request_id, error="Gateway not running")
        model_id = request.model_id
        local_models = self.local.models
        if model_id in local_models:
            local_replicas = self.local.replicas.get(model_id, {})
            local_available = sum(1 for r in local_replicas.values() if r.is_available)
            if local_available > 0:
                return await self.local.infer(request)
        node_id = self.lb.route_request(request)
        if not node_id:
            return InferenceResponse(
                request_id=request.request_id,
                error=f"Model {model_id} not available locally or on any peer",
            )
        request.node_id = node_id
        return await self.lb.submit_request(request)

    async def stream_serve(self, request: InferenceRequest):
        if not self._running:
            yield {"type": "error", "error": "Gateway not running"}
            return
        model_id = request.model_id
        if model_id in self.local.models:
            async for chunk in self.local.stream_infer(request):
                yield chunk
            return
        node_id = self.lb.route_request(request)
        if not node_id:
            yield {"type": "error", "error": f"Model {model_id} not available locally or on any peer"}
            return
        yield {"type": "error", "error": "Streaming across P2P nodes not yet supported; use local inference"}

    async def load_model(self, config: ModelServeConfig, weights: dict[str, np.ndarray] | None = None) -> ModelReplica:
        replica = await self.local.load_model(config, weights)
        self.lb.add_model_to_node(self.local.node_id, config.model_id or config.model_name)
        if config.mirror_enabled:
            for nid in self.lb.nodes:
                if nid != self.local.node_id:
                    self.lb.mirror.register_mirror(config.model_id or config.model_name, nid)
        return replica

    def get_status(self) -> dict[str, Any]:
        return {
            "gateway": "running" if self._running else "stopped",
            "local": self.local.get_status(),
            "cluster": self.lb.get_status(),
        }