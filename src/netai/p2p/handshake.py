"""P2P node handshake and capability exchange protocol.

Nodes advertise their compute/storage capabilities to peers, collect
peer info, and collaboratively determine optimal pipeline assignments.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import platform
import time
from typing import Any

import aiohttp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None

try:
    from importlib.metadata import version as _version
    NETAI_VERSION = _version("netai")
except Exception:
    NETAI_VERSION = "0.2.2"


class NodeCapabilities(BaseModel):
    node_id: str = ""
    hostname: str = Field(default_factory=platform.node)
    cpu_cores: int = Field(default_factory=lambda: os.cpu_count() or 1)
    cpu_available: int = 0
    ram_total_mb: float = 0.0
    ram_available_mb: float = 0.0
    gpu_name: str = ""
    gpu_vram_mb: float = 0.0
    gpu_count: int = 0
    cuda_available: bool = False
    rocm_available: bool = False
    disk_free_gb: float = 0.0
    network_mbps_down: float = 0.0
    network_mbps_up: float = 0.0
    python_version: str = Field(default_factory=platform.python_version)
    torch_version: str = ""
    netai_version: str = NETAI_VERSION
    supported_architectures: list[str] = Field(default_factory=lambda: ["gpt2"])
    max_batch_size: int = 1
    preferred_precision: str = "float32"
    current_load: int = 0
    uptime_s: float = 0.0
    last_updated: float = Field(default_factory=time.time)

    model_config = {"arbitrary_types_allowed": True}


class NodeScore(BaseModel):
    node_id: str = ""
    compute_score: float = 0.0
    memory_score: float = 0.0
    network_score: float = 0.0
    reliability_score: float = 0.0
    total_score: float = 0.0
    rank: int = 0

    model_config = {"arbitrary_types_allowed": True}


def detect_capabilities(node_id: str = "") -> NodeCapabilities:
    caps = NodeCapabilities(node_id=node_id or hashlib.sha256(os.urandom(8)).hexdigest()[:12])
    caps.cpu_available = os.cpu_count() or 1

    try:
        import psutil
        mem = psutil.virtual_memory()
        caps.ram_total_mb = mem.total / (1024 * 1024)
        caps.ram_available_mb = mem.available / (1024 * 1024)
        disk = psutil.disk_usage("/")
        caps.disk_free_gb = disk.free / (1024 * 1024 * 1024)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        caps.ram_total_mb = int(line.split()[1]) / 1024
                    elif "MemAvailable" in line:
                        caps.ram_available_mb = int(line.split()[1]) / 1024
        except Exception:
            caps.ram_total_mb = 8192
            caps.ram_available_mb = 4096

    if HAS_TORCH:
        caps.torch_version = torch.__version__
        caps.cuda_available = torch.cuda.is_available()
        if caps.cuda_available:
            caps.gpu_count = torch.cuda.device_count()
            caps.gpu_name = torch.cuda.get_device_name(0) if caps.gpu_count > 0 else ""
            caps.gpu_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    else:
        try:
            import subprocess
            result = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                caps.rocm_available = True
                caps.gpu_count = 1
                caps.gpu_name = "AMD Radeon (ROCm)"
        except Exception:
            pass

    return caps


class HandshakeProtocol:
    """Manages P2P handshakes, capability collection, and node ranking."""

    def __init__(self, node_id: str = "", port: int = 8001, host: str = "127.0.0.1"):
        self.node_id = node_id or hashlib.sha256(os.urandom(8)).hexdigest()[:12]
        self.port = port
        self.host = host
        self.capabilities = detect_capabilities(self.node_id)
        self.peer_capabilities: dict[str, NodeCapabilities] = {}
        self.peer_scores: dict[str, NodeScore] = {}
        self.peer_endpoints: dict[str, str] = {}
        self._start_time = time.time()

    async def advertise(self, target_endpoint: str) -> bool:
        """Send this node's capabilities to a peer."""
        self.capabilities.last_updated = time.time()
        self.capabilities.uptime_s = time.time() - self._start_time

        payload = self.capabilities.model_dump()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(f"{target_endpoint}/api/p2p/handshake", json=payload) as resp:
                    if resp.status == 200:
                        result = await resp.json(content_type=None)
                        if result.get("accepted"):
                            self.peer_endpoints[result.get("node_id", "")] = target_endpoint
                            logger.info("Handshake accepted by %s", target_endpoint)
                            return True
                    logger.warning("Handshake rejected by %s (status=%d)", target_endpoint, resp.status)
                    return False
        except Exception as e:
            logger.error("Handshake failed to %s: %s", target_endpoint, e)
            return False

    async def collect_peer_capabilities(self, target_endpoint: str) -> NodeCapabilities | None:
        """Fetch capabilities from a peer."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{target_endpoint}/api/p2p/capabilities/{self.node_id}") as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        caps = NodeCapabilities(**data)
                        self.peer_capabilities[caps.node_id] = caps
                        self.peer_endpoints[caps.node_id] = target_endpoint
                        logger.info("Collected capabilities from %s (%s)", caps.node_id, caps.gpu_name or "CPU")
                        return caps
        except Exception as e:
            logger.error("Failed to collect capabilities from %s: %s", target_endpoint, e)
        return None

    def receive_handshake(self, peer_caps: dict[str, Any]) -> NodeCapabilities:
        """Process an incoming handshake from a peer."""
        caps = NodeCapabilities(**peer_caps)
        self.peer_capabilities[caps.node_id] = caps
        self._score_node(caps)
        logger.info("Received handshake from %s (cpu=%d, gpu=%s)", caps.node_id, caps.cpu_cores, caps.gpu_name or "none")
        return caps

    def _score_node(self, caps: NodeCapabilities) -> NodeScore:
        score = NodeScore(node_id=caps.node_id)
        score.compute_score = min(100, caps.cpu_cores * 2 + caps.gpu_count * 40 + (caps.gpu_vram_mb / 1024) * 3)
        score.memory_score = min(100, caps.ram_available_mb / 16384 * 100)
        score.network_score = min(100, (caps.network_mbps_up + caps.network_mbps_down) / 20)
        age_penalty = min(1.0, max(0, (time.time() - caps.last_updated) / 3600))
        score.reliability_score = 100 * (1 - age_penalty * 0.5)
        score.total_score = (
            score.compute_score * 0.4 + score.memory_score * 0.25 +
            score.network_score * 0.20 + score.reliability_score * 0.15
        )
        self.peer_scores[caps.node_id] = score
        self._rerank()
        return score

    def _rerank(self) -> None:
        sorted_nodes = sorted(self.peer_scores.values(), key=lambda s: s.total_score, reverse=True)
        for i, s in enumerate(sorted_nodes):
            s.rank = i + 1

    def suggest_pipeline_role(self) -> dict[str, Any]:
        """Suggest what role this node should take in the pipeline."""
        caps = self.capabilities
        if caps.gpu_count > 0 and caps.gpu_vram_mb > 4096:
            role = "compute_heavy"
            suggested_layers = max(2, int(caps.gpu_vram_mb / 1024))
        elif caps.cpu_cores >= 8:
            role = "compute_medium"
            suggested_layers = max(1, int(caps.cpu_cores / 4))
        else:
            role = "compute_light"
            suggested_layers = 1
        return {
            "node_id": caps.node_id,
            "role": role,
            "suggested_layers": suggested_layers,
            "preferred_precision": caps.preferred_precision,
            "max_batch_size": caps.max_batch_size,
        }

    def best_node_for_layers(self, num_layers: int, memory_per_layer_mb: float) -> list[dict[str, Any]]:
        """Find best peers capable of handling the given layers."""
        candidates = []
        required_vram = num_layers * memory_per_layer_mb
        for nid, caps in self.peer_capabilities.items():
            available_vram = caps.ram_available_mb if caps.gpu_count == 0 else caps.gpu_vram_mb * 0.8
            if available_vram >= required_vram:
                score = self.peer_scores.get(nid, NodeScore(node_id=nid))
                candidates.append({
                    "node_id": nid, "total_score": score.total_score,
                    "rank": score.rank, "vram_available": available_vram,
                    "can_fit": True,
                })
        candidates.sort(key=lambda c: c["total_score"], reverse=True)
        return candidates

    async def ping(self, target_endpoint: str) -> tuple[bool, float]:
        """Ping a peer and measure latency."""
        t0 = time.time()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{target_endpoint}/api/p2p/ping") as resp:
                    lat = (time.time() - t0) * 1000
                    return resp.status == 200, lat
        except Exception:
            return False, -1

    def get_peer_list(self) -> list[dict[str, Any]]:
        return [
            {
                "node_id": nid, "score": self.peer_scores.get(nid, NodeScore()).total_score,
                "rank": self.peer_scores.get(nid, NodeScore()).rank,
                "endpoint": self.peer_endpoints.get(nid, ""),
            }
            for nid in self.peer_capabilities
        ]

    def get_status(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "capabilities": self.capabilities.model_dump(),
            "peers": len(self.peer_capabilities),
            "peer_list": self.get_peer_list(),
            "role_suggestion": self.suggest_pipeline_role(),
            "uptime_s": time.time() - self._start_time,
        }
