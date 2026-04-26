"""JackIn manager — one-command node registration and contribution.

`netai jack-in` should be as simple as:
    netai jack-in --model glm-5.1

This module handles the full flow:
1. Discover pool nodes (seed nodes, mDNS, bootstrap servers)
2. Register capabilities (GPU, VRAM, CPU)
3. Download assigned model shard
4. Start contributing compute
5. Earn inference credits via PPLNS
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from netai.compute_pool.pool import ComputePool, PoolNode, PoolNodeStatus
from netai.compute_pool.pipeline import PipelineConfig, PipelineStatus
from netai.compute_pool.share import ShareDifficulty
from netai.resource.profiler import ResourceProfiler

logger = logging.getLogger(__name__)


class JackInConfig(BaseModel):
    pool_url: str = "http://localhost:7999"
    model_id: str = ""
    node_name: str = ""
    gpu_only: bool = False
    max_vram_mb: float = 0.0
    max_cpu_cores: int = 0
    bandwidth_mbps: float = 0.0
    auto_start: bool = True
    heartbeat_interval_s: float = 30.0
    reconnect_attempts: int = 5
    reconnect_delay_s: float = 5.0


class JackInManager:
    def __init__(self, config: JackInConfig | None = None):
        self.config = config or JackInConfig()
        self.pool = ComputePool(pool_id="local")
        self._node: PoolNode | None = None
        self._local_node_id: str = uuid.uuid4().hex[:12]
        self._running = False
        self._profile_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._inference_credits: float = 0.0
        self._profile_cache: Any | None = None
        self._profile_cache_time: float = 0.0

    async def start(self) -> dict[str, Any]:
        try:
            self._running = True
            profile = self._get_profile()
            profile_time = time.time()

            self._node = PoolNode(
                node_id=self._local_node_id,
                status=PoolNodeStatus.CONNECTING,
                gpu_count=profile.gpu_count,
                gpu_names=profile.gpu_names,
                vram_total_mb=sum(profile.gpu_vram_mb) if profile.gpu_vram_mb else 0,
                vram_available_mb=sum(profile.gpu_available_vram_mb) if profile.gpu_available_vram_mb else 0,
                cpu_cores=profile.cpu_cores,
                ram_total_gb=profile.ram_total_gb,
                ram_available_gb=profile.ram_available_gb,
            )

            if self.config.max_vram_mb > 0:
                self._node.vram_available_mb = min(self._node.vram_available_mb, self.config.max_vram_mb)
            if self.config.max_cpu_cores > 0:
                self._node.cpu_cores = min(self._node.cpu_cores, self.config.max_cpu_cores)

            if self.config.gpu_only and self._node.gpu_count == 0:
                logger.warning("GPU-only mode set but no GPU detected")

            result = await self.pool.jack_in(self._node, model_id=self.config.model_id)
            self._node = self.pool.nodes.get(self._local_node_id, self._node)

            await self.pool.start()

            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info(
                "Jacked in as %s (GPUs=%d, VRAM=%.1fGB, Credits=%.0f)",
                self._local_node_id,
                self._node.gpu_count,
                self._node.vram_available_mb / 1024,
                self._inference_credits,
            )
            return result
        except Exception as e:
            logger.error("Jack-in failed: %s", e)
            self._running = False
            raise

    async def stop(self):
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        await self.pool.jack_out(self._local_node_id)
        await self.pool.stop()
        logger.info("Jacked out node %s", self._local_node_id)

    def _get_profile(self):
        now = time.time()
        if self._profile_cache is not None and (now - self._profile_cache_time) < 300:
            return self._profile_cache
        profiler = ResourceProfiler()
        self._profile_cache = profiler.profile()
        self._profile_cache_time = now
        return self._profile_cache

    async def _heartbeat_loop(self):
        while self._running:
            try:
                self.pool.update_heartbeat(self._local_node_id)
                self._inference_credits = self.pool.get_inference_credits(self._local_node_id)
                if self._node and self.config.max_vram_mb == 0:
                    if (time.time() - self._profile_cache_time) > 300:
                        profile = self._get_profile()
                        self._node.vram_available_mb = sum(profile.gpu_available_vram_mb) if profile.gpu_available_vram_mb else 0
                await asyncio.sleep(self.config.heartbeat_interval_s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error: %s", e)
                await asyncio.sleep(self.config.heartbeat_interval_s)

    def get_local_status(self) -> dict[str, Any]:
        if not self._node:
            return {"status": "not_joined"}
        contribution = self.pool.get_contribution(self._local_node_id)
        plan = self.pool.orchestrator.get_pipeline(self.config.model_id) if self.config.model_id else None
        return {
            "node_id": self._node.node_id,
            "status": self._node.status.value,
            "gpus": self._node.gpu_count,
            "vram_gb": round(self._node.vram_total_mb / 1024, 1),
            "vram_available_gb": round(self._node.vram_available_mb / 1024, 1),
            "cpu_cores": self._node.cpu_cores,
            "model": self._node.current_model,
            "stage": self._node.current_stage,
            "inferences": self._node.inference_count,
            "inference_credits": self._inference_credits,
            "shares": contribution.total_shares if contribution else 0,
            "reliability": contribution.reliability if contribution else 0.0,
            "pipeline_ready": plan.is_complete if plan else False,
        }

    def get_pool_status(self) -> dict[str, Any]:
        status = self.pool.get_status()
        return status.model_dump()