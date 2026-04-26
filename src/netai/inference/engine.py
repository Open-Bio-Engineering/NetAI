"""Distributed inference engine - model serving, sharding, mirroring, request routing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from collections import deque
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class InferenceStatus(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    SERVING = "serving"
    DRAINING = "draining"
    ERROR = "error"
    OFFLINE = "offline"


class ShardType(str, Enum):
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    MOE_EXPERT = "moe_expert"
    REPLICA = "replica"


class InferenceRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    model_id: str = ""
    prompt: str = ""
    inputs: list[Any] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    max_tokens: int = Field(default=256, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=1000)
    stream: bool = False
    priority: int = Field(default=0, ge=0, le=10)
    timeout_ms: int = Field(default=30000, ge=1000, le=300000)
    user_id: str = ""
    group_id: str = ""
    node_id: str = ""
    created_at: float = Field(default_factory=time.time)


class InferenceResponse(BaseModel):
    request_id: str = ""
    model_id: str = ""
    outputs: list[Any] = Field(default_factory=list)
    text: str = ""
    tokens_generated: int = 0
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    node_id: str = ""
    finish_reason: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    error: str | None = None


class ModelShard(BaseModel):
    shard_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_id: str = ""
    shard_type: ShardType = ShardType.TENSOR
    shard_index: int = 0
    total_shards: int = 1
    layers: list[str] = Field(default_factory=list)
    size_mb: float = 0.0
    checksum: str = ""
    node_id: str = ""
    status: InferenceStatus = InferenceStatus.IDLE
    loaded_at: float = 0.0
    inference_count: int = 0
    avg_latency_ms: float = 0.0


class ModelReplica(BaseModel):
    replica_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_id: str = ""
    version: str = ""
    node_id: str = ""
    status: InferenceStatus = InferenceStatus.IDLE
    shard_ids: list[str] = Field(default_factory=list)
    loaded_at: float = 0.0
    inference_count: int = 0
    total_latency_ms: float = 0.0
    last_inference: float = 0.0
    capacity: int = 100
    current_load: int = 0
    health_score: float = 1.0

    @property
    def is_available(self) -> bool:
        return self.status in (InferenceStatus.READY, InferenceStatus.SERVING) and self.current_load < self.capacity

    @property
    def avg_latency(self) -> float:
        return self.total_latency_ms / max(self.inference_count, 1)

    @property
    def load_factor(self) -> float:
        return self.current_load / max(self.capacity, 1)


class ModelServeConfig(BaseModel):
    model_id: str = ""
    model_name: str = ""
    version: str = "latest"
    num_replicas: int = 1
    num_shards: int = 1
    shard_type: ShardType = ShardType.TENSOR
    mirror_enabled: bool = True
    max_batch_size: int = 32
    max_sequence_length: int = 2048
    quantization: str = ""
    device: str = "auto"
    group_id: str = ""
    cache_size_mb: int = 512
    timeout_ms: int = 30000


class InferenceMetrics(BaseModel):
    request_id: str = ""
    model_id: str = ""
    node_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    batch_size: int = 1
    gpu_util: float = 0.0
    gpu_mem_mb: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class InferenceEngine:
    def __init__(self, node_id: str = "", cache_dir: str = "./inference_cache"):
        self.node_id = node_id or uuid.uuid4().hex[:12]
        self.cache_dir = cache_dir
        self.status = InferenceStatus.IDLE
        self.models: dict[str, ModelServeConfig] = {}
        self.replicas: dict[str, dict[str, ModelReplica]] = {}
        self.shards: dict[str, dict[str, ModelShard]] = {}
        self._model_weights: dict[str, dict[str, np.ndarray]] = {}
        self._request_queue: asyncio.Queue | None = None
        self._response_futures: dict[str, asyncio.Future] = {}
        self.metrics_history: deque[InferenceMetrics] = deque(maxlen=10000)
        self._inference_task: asyncio.Task | None = None
        self._running = False
        self._lock = asyncio.Lock()
        os.makedirs(cache_dir, exist_ok=True)

    async def start(self):
        self._request_queue = asyncio.Queue(maxsize=1000)
        self._running = True
        self.status = InferenceStatus.READY
        self._inference_task = asyncio.create_task(self._inference_loop())
        logger.info("Inference engine started on node %s", self.node_id)

    async def stop(self):
        self._running = False
        self.status = InferenceStatus.OFFLINE
        if self._inference_task and not self._inference_task.done():
            self._inference_task.cancel()

    async def load_model(self, config: ModelServeConfig, weights: dict[str, np.ndarray] | None = None) -> ModelReplica:
        self.status = InferenceStatus.LOADING
        model_id = config.model_id or config.model_name
        config.model_id = model_id
        self.models[model_id] = config

        if weights is None:
            weights = self._generate_dummy_weights(config)
        self._model_weights[model_id] = weights

        replica = ModelReplica(
            model_id=model_id,
            version=config.version,
            node_id=self.node_id,
            status=InferenceStatus.READY,
        )

        if config.num_shards > 1:
            shard_map = await self._create_shards(config, weights)
            replica.shard_ids = list(shard_map.keys())
            self.shards[model_id] = shard_map
        else:
            shard = ModelShard(
                model_id=model_id,
                shard_type=ShardType.REPLICA,
                shard_index=0,
                total_shards=1,
                status=InferenceStatus.READY,
                node_id=self.node_id,
            )
            self.shards.setdefault(model_id, {})[shard.shard_id] = shard
            replica.shard_ids = [shard.shard_id]

        replica.loaded_at = time.time()
        self.replicas.setdefault(model_id, {})[replica.replica_id] = replica
        self.status = InferenceStatus.READY
        logger.info("Model %s loaded (shards=%d, replica=%s)", model_id, config.num_shards, replica.replica_id)
        return replica

    async def _create_shards(self, config: ModelServeConfig, weights: dict[str, np.ndarray]) -> dict[str, ModelShard]:
        shard_map = {}
        layers = sorted(weights.keys())
        layers_per_shard = max(1, len(layers) // config.num_shards)
        for i in range(config.num_shards):
            start = i * layers_per_shard
            end = min(start + layers_per_shard, len(layers)) if i < config.num_shards - 1 else len(layers)
            shard = ModelShard(
                model_id=config.model_id,
                shard_type=config.shard_type,
                shard_index=i,
                total_shards=config.num_shards,
                layers=layers[start:end],
                status=InferenceStatus.READY,
                node_id=self.node_id,
            )
            total_size = sum(weights[k].nbytes for k in layers[start:end] if k in weights)
            shard.size_mb = total_size / (1024 * 1024)
            shard_map[shard.shard_id] = shard
        return shard_map

    def _generate_dummy_weights(self, config: ModelServeConfig) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(42)
        layers = []
        for i in range(12):
            layers.append(f"layer_{i}.attn.q_weight")
            layers.append(f"layer_{i}.attn.k_weight")
            layers.append(f"layer_{i}.attn.v_weight")
            layers.append(f"layer_{i}.ffn.up_weight")
        layers.extend(["embed.weight", "output.weight"])
        weights = {}
        for name in layers:
            shape = (768, 768) if "layer" in name else (50257, 768)
            weights[name] = rng.standard_normal(shape).astype(np.float32) * 0.02
        return weights

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        if not self._running:
            return InferenceResponse(request_id=request.request_id, error="Engine not running")
        model_id = request.model_id
        if model_id not in self._model_weights:
            return InferenceResponse(request_id=request.request_id, error=f"Model {model_id} not loaded")
        if len(request.prompt) > 100000:
            return InferenceResponse(request_id=request.request_id, error="Prompt too long (max 100K chars)")

        t0 = time.time()

        selected_rep = None
        available = [(rid, rep) for rid, rep in self.replicas.get(model_id, {}).items() if rep.is_available]
        if available:
            available.sort(key=lambda x: x[1].current_load)
            rid, selected_rep = available[0]
            selected_rep.current_load += 1

        try:
            try:
                result = await asyncio.wait_for(
                    self._run_inference(request),
                    timeout=request.timeout_ms / 1000.0,
                )
            except asyncio.TimeoutError:
                return InferenceResponse(request_id=request.request_id, error="Inference timed out")
            elapsed = (time.time() - t0) * 1000
            result.latency_ms = elapsed
            result.node_id = self.node_id
            if elapsed > 0 and result.tokens_generated > 0:
                result.tokens_per_second = result.tokens_generated / (elapsed / 1000)
            metrics = InferenceMetrics(
                request_id=request.request_id,
                model_id=model_id,
                node_id=self.node_id,
                prompt_tokens=len(request.prompt) // 4 + 1,
                completion_tokens=result.tokens_generated,
                latency_ms=elapsed,
                tokens_per_second=result.tokens_per_second,
                batch_size=1,
            )
            self._record_metrics(metrics)
            if selected_rep is not None:
                selected_rep.inference_count += 1
                selected_rep.total_latency_ms += elapsed
                selected_rep.last_inference = time.time()
                selected_rep.current_load = max(0, selected_rep.current_load - 1)
                if selected_rep.current_load == 0:
                    selected_rep.status = InferenceStatus.READY
            return result
        except Exception as e:
            if selected_rep is not None:
                selected_rep.current_load = max(0, selected_rep.current_load - 1)
                if selected_rep.current_load == 0:
                    selected_rep.status = InferenceStatus.READY
            return InferenceResponse(request_id=request.request_id, error=str(e))

    async def _run_inference(self, request: InferenceRequest) -> InferenceResponse:
        await asyncio.sleep(0.001)
        rng = np.random.default_rng(hash(request.prompt) % (2**31))
        num_tokens = min(request.max_tokens, rng.integers(10, 50))
        output_tokens = rng.integers(0, 1000, size=num_tokens).tolist()
        words = ["the", "model", "predicts", "this", "output", "based", "on",
                 "input", "data", "distributed", "across", "nodes"]
        generated = " ".join(rng.choice(words, size=min(num_tokens, 20)))
        return InferenceResponse(
            request_id=request.request_id,
            model_id=request.model_id,
            outputs=output_tokens,
            text=generated,
            tokens_generated=num_tokens,
            finish_reason="stop" if num_tokens < request.max_tokens else "length",
            usage={"prompt_tokens": max(1, len(request.prompt) // 4), "completion_tokens": num_tokens, "total_tokens": max(1, len(request.prompt) // 4) + num_tokens},
        )

    async def stream_infer(self, request: InferenceRequest):
        if not self._running:
            yield {"type": "error", "error": "Engine not running"}
            return
        model_id = request.model_id
        if model_id not in self._model_weights:
            yield {"type": "error", "error": f"Model {model_id} not loaded"}
            return

        t0 = time.time()

        rng = np.random.default_rng(int(hashlib.sha256(request.request_id.encode()).hexdigest()[:8], 16))
        num_tokens = min(request.max_tokens, rng.integers(10, 50))
        words = ["the", "model", "predicts", "this", "output", "based", "on",
                 "input", "data", "distributed", "across", "nodes"]

        request_id = request.request_id

        try:
            yield {
                "type": "start",
                "request_id": request_id,
                "model_id": model_id,
                "node_id": self.node_id,
            }

            generated_words = []
            for i in range(num_tokens):
                await asyncio.sleep(0.01)
                token_idx = int(rng.integers(0, 1000))
                word = str(rng.choice(words))
                generated_words.append(word)
                yield {
                    "type": "token",
                    "token_id": token_idx,
                    "text": word if i < min(num_tokens, 20) else "",
                    "index": i,
                    "request_id": request_id,
                }

            elapsed_ms = (time.time() - t0) * 1000
            generated_text = " ".join(generated_words[:min(num_tokens, 20)])

            metrics = InferenceMetrics(
                request_id=request_id,
                model_id=model_id,
                node_id=self.node_id,
                completion_tokens=num_tokens,
                latency_ms=elapsed_ms,
                tokens_per_second=num_tokens / max(elapsed_ms / 1000, 0.001),
                batch_size=1,
            )
            self._record_metrics(metrics)

            yield {
                "type": "done",
                "request_id": request_id,
                "text": generated_text,
                "tokens_generated": num_tokens,
                "latency_ms": round(elapsed_ms, 2),
                "tokens_per_second": round(metrics.tokens_per_second, 2),
                "finish_reason": "stop" if num_tokens < request.max_tokens else "length",
                "usage": {
                    "prompt_tokens": max(1, len(request.prompt) // 4),
                    "completion_tokens": num_tokens,
                    "total_tokens": max(1, len(request.prompt) // 4) + num_tokens,
                },
            }
        except Exception as e:
            yield {"type": "error", "error": str(e)}

    async def _inference_loop(self):
        while self._running:
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Inference loop error: %s", e)

    def _record_metrics(self, metrics: InferenceMetrics):
        self.metrics_history.append(metrics)

    async def drain_model(self, model_id: str, timeout: float = 30.0) -> bool:
        if model_id not in self.replicas:
            return True
        replicas = self.replicas[model_id]
        for rid, rep in replicas.items():
            rep.status = InferenceStatus.DRAINING
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(rep.current_load == 0 for rep in replicas.values()):
                return True
            await asyncio.sleep(0.1)
        logger.warning("Drain timed out for model %s, forcing unload", model_id)
        return False

    async def unload_model(self, model_id: str, drain_timeout: float = 30.0) -> bool:
        await self.drain_model(model_id, drain_timeout)
        if model_id in self._model_weights:
            del self._model_weights[model_id]
        self.models.pop(model_id, None)
        self.replicas.pop(model_id, None)
        self.shards.pop(model_id, None)
        logger.info("Model %s unloaded", model_id)
        return True

    def get_status(self) -> dict[str, Any]:
        models_info = {}
        for mid, config in self.models.items():
            reps = self.replicas.get(mid, {})
            available = sum(1 for r in reps.values() if r.is_available)
            serving = sum(1 for r in reps.values() if r.status == InferenceStatus.SERVING)
            total_inferences = sum(r.inference_count for r in reps.values())
            avg_lat = np.mean([r.avg_latency for r in reps.values()]) if reps else 0
            models_info[mid] = {
                "name": config.model_name,
                "version": config.version,
                "shards": config.num_shards,
                "replicas": len(reps),
                "available_replicas": available,
                "serving_replicas": serving,
                "total_inferences": total_inferences,
                "avg_latency_ms": round(avg_lat, 2),
            }
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "models_loaded": len(self.models),
            "models": models_info,
            "total_inferences": len(self.metrics_history),
            "cache_dir": self.cache_dir,
        }


class ModelMirror:
    def __init__(self, cache_dir: str = "./inference_cache"):
        self.cache_dir = cache_dir
        self.mirrors: dict[str, list[str]] = {}
        self._sync_tasks: dict[str, asyncio.Task] = {}
        os.makedirs(cache_dir, exist_ok=True)

    def register_mirror(self, model_id: str, node_id: str) -> bool:
        if model_id not in self.mirrors:
            self.mirrors[model_id] = []
        if node_id not in self.mirrors[model_id]:
            self.mirrors[model_id].append(node_id)
            logger.info("Mirror registered: model=%s node=%s (total mirrors=%d)",
                       model_id, node_id, len(self.mirrors[model_id]))
            return True
        return False

    def unregister_mirror(self, model_id: str, node_id: str):
        if model_id in self.mirrors:
            self.mirrors[model_id] = [n for n in self.mirrors[model_id] if n != node_id]

    def get_mirrors(self, model_id: str) -> list[str]:
        return self.mirrors.get(model_id, [])

    def get_mirror_count(self, model_id: str) -> int:
        return len(self.mirrors.get(model_id, []))

    def find_nearest_mirror(self, model_id: str, requesting_node: str, node_latencies: dict[str, float] | None = None) -> str | None:
        mirrors = self.mirrors.get(model_id, [])
        if not mirrors:
            return None
        if requesting_node in mirrors:
            return requesting_node
        if node_latencies:
            mirrors_with_latency = [(n, node_latencies.get(n, 999)) for n in mirrors]
            mirrors_with_latency.sort(key=lambda x: x[1])
            return mirrors_with_latency[0][0]
        return mirrors[0]

    async def sync_model_to_node(
        self,
        model_id: str,
        weights: dict[str, np.ndarray],
        target_node_endpoint: str,
    ) -> bool:
        path = os.path.join(self.cache_dir, f"mirror_{model_id}_{int(time.time())}.npz")
        np.savez_compressed(path, **weights)
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                data = {"model_id": model_id, "size_mb": os.path.getsize(path) / (1024*1024)}
                async with session.post(
                    f"{target_node_endpoint}/api/inference/load",
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.error("Mirror sync failed: %s", e)
            return False
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def get_status(self) -> dict[str, Any]:
        return {
            "mirrored_models": len(self.mirrors),
            "models": {
                mid: {"mirrors": nodes, "count": len(nodes)}
                for mid, nodes in self.mirrors.items()
            },
        }