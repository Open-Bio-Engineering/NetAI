"""Pipeline-parallel distributed executor — orchestrates inference across P2P nodes.

Each node runs a contiguous slice of transformer layers. The PipelineExecutor
coordinates: assigning layers to nodes, sending activations between stages,
collecting final logits, and handling node failures with reassignment.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from netai.inference.native_engine import NativeInferenceEngine, LayerResult, TransformerConfig

logger = logging.getLogger(__name__)


class PipelineStage(BaseModel):
    stage_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    node_id: str = ""
    model_id: str = ""
    stage_index: int = 0
    total_stages: int = 0
    layer_start: int = 0
    layer_end: int = 0
    num_layers: int = 0
    vram_mb: float = 0.0
    status: str = "idle"
    loaded_at: float = 0.0
    inference_count: int = 0
    total_latency_ms: float = 0.0
    last_heartbeat: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.inference_count, 1)


class PipelineResult(BaseModel):
    request_id: str = ""
    model_id: str = ""
    prompt_tokens: list[int] = Field(default_factory=list)
    generated_tokens: list[int] = Field(default_factory=list)
    text: str = ""
    total_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    num_stages: int = 0
    stage_latencies: list[float] = Field(default_factory=list)
    finish_reason: str = "stop"
    error: str = ""


class PipelineExecutor:
    """Orchestrates distributed pipeline-parallel inference across P2P nodes.

    Assigns transformer layers to nodes based on their available VRAM,
    sends activation tensors between stages, and produces final output.
    Handles node failures by re-assigning layers.
    """

    def __init__(self, local_engine: NativeInferenceEngine | None = None):
        self.local_engine = local_engine or NativeInferenceEngine()
        self.pipelines: dict[str, dict[str, PipelineStage]] = {}
        self.configs: dict[str, TransformerConfig] = {}
        self._results_buffer: dict[str, dict[str, Any]] = {}
        self._pending_requests: dict[str, asyncio.Future] = {}

    def plan_pipeline(
        self,
        model_id: str,
        config: TransformerConfig,
        node_resources: list[dict[str, Any]],
    ) -> list[PipelineStage]:
        """Divide model layers across available nodes based on their VRAM."""

        if config.num_layers <= 0:
            return []

        sorted_nodes = sorted(node_resources, key=lambda n: n.get("vram_available_mb", 0), reverse=True)
        total_vram = sum(n.get("vram_available_mb", 0) for n in sorted_nodes)
        model_vram = config.vram_mb(bytes_per_param=2, batch_size=1, seq_len=config.max_position_embeddings)

        bytes_per_layer = model_vram / max(config.num_layers, 1)
        stages = []
        layers_remaining = list(range(config.num_layers))

        for node in sorted_nodes:
            if not layers_remaining:
                break
            node_vram = node.get("vram_available_mb", 0) * 0.8
            node_id = node.get("node_id", "")
            max_layers = int(node_vram / max(bytes_per_layer, 0.1))
            max_layers = max(1, min(max_layers, len(layers_remaining)))

            stage_layers = layers_remaining[:max_layers]
            stage = PipelineStage(
                node_id=node_id,
                model_id=model_id,
                stage_index=len(stages),
                total_stages=0,
                layer_start=stage_layers[0],
                layer_end=stage_layers[-1],
                num_layers=len(stage_layers),
                vram_mb=bytes_per_layer * len(stage_layers) * 1.1,
                status="assigned",
            )
            stages.append(stage)
            layers_remaining = layers_remaining[max_layers:]

        if layers_remaining:
            for i, layer_idx in enumerate(layers_remaining):
                stage = PipelineStage(
                    node_id="overflow",
                    model_id=model_id,
                    stage_index=len(stages),
                    total_stages=0,
                    layer_start=layer_idx,
                    layer_end=layer_idx,
                    num_layers=1,
                    vram_mb=bytes_per_layer * 1.1,
                    status="pending",
                )
                stages.append(stage)

        for s in stages:
            s.total_stages = len(stages)

        self.pipelines[model_id] = {s.stage_id: s for s in stages}
        self.configs[model_id] = config
        logger.info("Pipeline planned: %d stages for %s (%d layers)",
                     len(stages), model_id, config.num_layers)
        return stages

    def assign_local_stages(self, model_id: str) -> list[PipelineStage]:
        """Assign local-engine stages (run on this node) for a given model."""
        stages = self.pipelines.get(model_id, {})
        local_stages = []
        for stage in stages.values():
            if stage.node_id == self.local_engine.node_id or stage.node_id == "local":
                stage.node_id = self.local_engine.node_id
                stage.status = "ready"
                local_stages.append(stage)
        return local_stages

    async def run_local_stage(
        self,
        model_id: str,
        stage: PipelineStage,
        hidden: np.ndarray,
        request_id: str = "",
    ) -> tuple[np.ndarray, LayerResult]:
        """Run a pipeline stage locally through the NativeInferenceEngine."""
        output_hidden = self.local_engine.forward(
            hidden, model_id, stage.layer_start, stage.layer_end
        )
        result = self.local_engine.forward_segment(
            hidden=hidden,
            model_id=model_id,
            layer_start=stage.layer_start,
            layer_end=stage.layer_end,
            request_id=request_id,
        )
        return output_hidden, result

    async def send_activation(
        self,
        target_endpoint: str,
        hidden: np.ndarray,
        request_id: str,
        stage_index: int,
        model_id: str,
    ) -> np.ndarray | None:
        """Send activation tensor to a remote P2P node for the next pipeline stage."""
        import aiohttp
        shape = list(hidden.shape)
        dtype = str(hidden.dtype)
        data = hidden.astype(np.float32).tobytes()
        payload = {
            "request_id": request_id,
            "model_id": model_id,
            "stage_index": stage_index,
            "shape": shape,
            "dtype": dtype,
            "data_hex": data.hex(),
            "data_size": len(data),
        }
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
                url = f"{target_endpoint}/api/inference/pipeline/activate"
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        logger.error("Activation send failed to %s: %d", target_endpoint, resp.status)
                        return None
                    result = await resp.json(content_type=None)
                    if "error" in result:
                        logger.error("Activation error from %s: %s", target_endpoint, result["error"])
                        return None
                    result_data_hex = result.get("data_hex")
                    result_shape = result.get("shape", shape)
                    result_dtype = result.get("dtype", dtype)
                    if not result_data_hex:
                        logger.error("Remote node %s returned no activation data", target_endpoint)
                        return None
                    return np.frombuffer(
                        bytes.fromhex(result_data_hex),
                        dtype=np.dtype(result_dtype),
                    ).reshape(result_shape)
        except Exception as e:
            logger.error("Failed to send activation to %s: %s", target_endpoint, e)
            return None

    async def run_pipeline(
        self,
        model_id: str,
        prompt_tokens: list[int],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        node_endpoints: dict[str, str] | None = None,
    ) -> PipelineResult:
        """Run full pipeline-parallel inference across all stages."""
        request_id = uuid.uuid4().hex[:16]
        config = self.configs.get(model_id)
        if config is None:
            return PipelineResult(request_id=request_id, model_id=model_id, error=f"Model {model_id} not configured")

        stages_dict = self.pipelines.get(model_id)
        if not stages_dict:
            return PipelineResult(request_id=request_id, model_id=model_id, error=f"No pipeline for {model_id}")

        stages = sorted(stages_dict.values(), key=lambda s: s.stage_index)
        t0 = time.time()
        stage_latencies = []

        embed = self.local_engine.embed_tokens.get(model_id)
        if embed is None:
            return PipelineResult(request_id=request_id, model_id=model_id, error=f"No embedding for {model_id}")

        try:
            input_ids = np.array(prompt_tokens, dtype=np.int64).reshape(1, -1)
            hidden = embed[input_ids].astype(np.float32)

            for stage in stages:
                stage_t0 = time.time()
                local = (stage.node_id == self.local_engine.node_id or
                         stage.node_id == "local" or
                         stage.node_id == "overflow" or
                         (node_endpoints is not None and stage.node_id not in node_endpoints))

                if local:
                    hidden, layer_result = await self.run_local_stage(
                        model_id, stage, hidden, request_id
                    )
                    stage.inference_count += 1
                    stage_lat = (time.time() - stage_t0) * 1000
                    stage.total_latency_ms += stage_lat
                    stage.last_heartbeat = time.time()
                    stage_latencies.append(stage_lat)
                elif node_endpoints and stage.node_id in node_endpoints:
                    endpoint = node_endpoints[stage.node_id]
                    result = await self.send_activation(
                        endpoint, hidden, request_id, stage.stage_index, model_id
                    )
                    if result is None:
                        logger.warning("Remote stage %d failed, running locally", stage.stage_index)
                        hidden, layer_result = await self.run_local_stage(
                            model_id, stage, hidden, request_id
                        )
                        stage_lat = (time.time() - stage_t0) * 1000
                        stage_latencies.append(stage_lat)
                    else:
                        hidden = result
                        stage_lat = (time.time() - stage_t0) * 1000
                        stage_latencies.append(stage_lat)
                else:
                    hidden, _ = await self.run_local_stage(
                        model_id, stage, hidden, request_id
                    )
                    stage_lat = (time.time() - stage_t0) * 1000
                    stage_latencies.append(stage_lat)

            ln_f = self.local_engine.layer_norm_f.get(model_id)
            if ln_f is not None:
                hidden = self._layer_norm(hidden[:, -1, :], ln_f[0], ln_f[1], config.layer_norm_eps)
            else:
                hidden = hidden[:, -1, :]

            output_w = self.local_engine.output_proj.get(model_id)
            if output_w is not None:
                logits = hidden @ output_w.T
            else:
                logits = hidden @ embed.T

            logits = logits / max(temperature, 0.01)
            if top_k > 0 and top_k < logits.shape[-1]:
                top_k_indices = np.argsort(logits[0])[-top_k:]
                mask = np.full(logits.shape, -1e9, dtype=np.float32)
                mask[0, top_k_indices] = logits[0, top_k_indices]
                logits = mask
            from netai.inference.native_engine import _softmax
            probs = _softmax(logits, axis=-1)[0]
            next_token = int(np.random.choice(len(probs), p=probs))

            total_latency = (time.time() - t0) * 1000
            tps = 1.0 / max(total_latency / 1000, 0.001)

            return PipelineResult(
                request_id=request_id,
                model_id=model_id,
                prompt_tokens=prompt_tokens,
                generated_tokens=[next_token],
                total_latency_ms=round(total_latency, 1),
                tokens_per_second=round(tps, 1),
                num_stages=len(stages),
                stage_latencies=[round(s, 1) for s in stage_latencies],
            )

        except Exception as e:
            logger.error("Pipeline execution failed: %s", e)
            return PipelineResult(
                request_id=request_id, model_id=model_id,
                error=str(e), total_latency_ms=(time.time() - t0) * 1000,
            )

    async def generate_autoregressive(
        self,
        model_id: str,
        prompt_tokens: list[int],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        node_endpoints: dict[str, str] | None = None,
    ) -> PipelineResult:
        """Autoregressive generation: run pipeline once per token."""
        from netai.inference.native_engine import _softmax

        request_id = uuid.uuid4().hex[:16]
        config = self.configs.get(model_id)
        if config is None:
            return PipelineResult(request_id=request_id, model_id=model_id,
                                   error=f"Model {model_id} not configured")

        stages_dict = self.pipelines.get(model_id)
        if not stages_dict:
            return PipelineResult(request_id=request_id, model_id=model_id,
                                   error=f"No pipeline for {model_id}")

        stages = sorted(stages_dict.values(), key=lambda s: s.stage_index)
        embed = self.local_engine.embed_tokens.get(model_id)
        if embed is None:
            return PipelineResult(request_id=request_id, model_id=model_id,
                                   error=f"No embedding for {model_id}")

        t0 = time.time()
        tokens = list(prompt_tokens)
        generated = []

        for step in range(max_tokens):
            input_ids = np.array(tokens, dtype=np.int64).reshape(1, -1)
            hidden = embed[input_ids].astype(np.float32)

            for stage in stages:
                hidden = self.local_engine.forward(hidden, model_id, stage.layer_start, stage.layer_end)

            ln_f = self.local_engine.layer_norm_f.get(model_id)
            if ln_f is not None:
                last_hidden = self._layer_norm(hidden[:, -1, :], ln_f[0], ln_f[1], config.layer_norm_eps)
            else:
                last_hidden = hidden[:, -1, :]

            output_w = self.local_engine.output_proj.get(model_id)
            logits = last_hidden @ (output_w.T if output_w is not None else embed.T)
            logits = logits / max(temperature, 0.01)

            if top_k > 0 and top_k < logits.shape[-1]:
                top_k_indices = np.argsort(logits[0])[-top_k:]
                mask = np.full(logits.shape, -1e9, dtype=np.float32)
                mask[0, top_k_indices] = logits[0, top_k_indices]
                logits = mask

            probs = _softmax(logits, axis=-1)[0]
            next_token = int(np.random.choice(len(probs), p=probs))
            tokens.append(next_token)
            generated.append(next_token)

        total_latency = (time.time() - t0) * 1000
        tps = len(generated) / max(total_latency / 1000, 0.001)

        return PipelineResult(
            request_id=request_id,
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated,
            total_latency_ms=round(total_latency, 1),
            tokens_per_second=round(tps, 1),
            num_stages=len(stages),
        )

    def _layer_norm(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return weight * (x - mean) / np.sqrt(var + eps) + bias

    def remove_pipeline(self, model_id: str) -> bool:
        self.pipelines.pop(model_id, None)
        self.configs.pop(model_id, None)
        return True

    def get_pipeline_status(self, model_id: str) -> dict[str, Any]:
        stages = self.pipelines.get(model_id, {})
        if not stages:
            return {"model_id": model_id, "stages": 0, "status": "not_found"}
        stage_list = sorted(stages.values(), key=lambda s: s.stage_index)
        return {
            "model_id": model_id,
            "stages": len(stage_list),
            "status": "ready" if all(s.status in ("ready", "assigned") for s in stage_list) else "partial",
            "stage_details": [
                {
                    "stage_id": s.stage_id,
                    "stage_index": s.stage_index,
                    "node_id": s.node_id,
                    "layers": f"{s.layer_start}-{s.layer_end}",
                    "num_layers": s.num_layers,
                    "status": s.status,
                    "inferences": s.inference_count,
                    "avg_latency_ms": round(s.avg_latency_ms, 1),
                }
                for s in stage_list
            ],
            "total_layers": sum(s.num_layers for s in stage_list),
            "total_inferences": sum(s.inference_count for s in stage_list),
        }

    def list_pipelines(self) -> list[dict[str, Any]]:
        return [self.get_pipeline_status(mid) for mid in self.pipelines]