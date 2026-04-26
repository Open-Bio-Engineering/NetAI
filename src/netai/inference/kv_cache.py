"""KV-cache management for distributed inference across P2P nodes."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from collections import OrderedDict, deque
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CacheEntry(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    cache_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_id: str = ""
    request_hash: str = ""
    prompt_tokens: list[int] = Field(default_factory=list)
    kv_tensors: Any = None
    num_layers: int = 0
    num_heads: int = 0
    head_dim: int = 0
    seq_length: int = 0
    size_mb: float = 0.0
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = 0
    node_id: str = ""
    is_partial: bool = False
    partition_id: str = ""


class CachePartition(BaseModel):
    partition_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    node_id: str = ""
    model_id: str = ""
    layer_start: int = 0
    layer_end: int = 0
    entries: int = 0
    size_mb: float = 0.0
    max_size_mb: float = 512.0

    @property
    def usage_pct(self) -> float:
        return (self.size_mb / self.max_size_mb * 100) if self.max_size_mb > 0 else 0


class KVCacheManager:
    def __init__(self, max_size_mb: float = 2048.0, ttl_seconds: float = 600.0):
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._partitions: dict[str, dict[str, CachePartition]] = {}
        self._current_size_mb: float = 0.0
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._thread_lock = threading.Lock()

    def _compute_request_hash(self, model_id: str, prompt: str, params: dict[str, Any]) -> str:
        raw = f"{model_id}:{prompt}:{sorted(params.items())}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, model_id: str, prompt: str, params: dict[str, Any] | None = None) -> CacheEntry | None:
        with self._thread_lock:
            req_hash = self._compute_request_hash(model_id, prompt, params or {})
            entry = self._cache.get(req_hash)
            if entry is None:
                self._misses += 1
                return None
            if time.time() - entry.last_accessed > self.ttl_seconds:
                self._evict(req_hash)
                self._misses += 1
                return None
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._cache.move_to_end(req_hash)
            self._hits += 1
            return entry

    def put(self, model_id: str, prompt: str, kv_data: Any,
            params: dict[str, Any] | None = None) -> CacheEntry:
        with self._thread_lock:
            req_hash = self._compute_request_hash(model_id, prompt, params or {})
            num_layers = 0
            seq_length = 0
            total_elements = 0
            if isinstance(kv_data, dict):
                num_layers = len(kv_data)
                for v in kv_data.values():
                    if isinstance(v, np.ndarray):
                        total_elements += v.size
                    elif isinstance(v, (list, tuple)):
                        total_elements += len(v)
                    else:
                        total_elements += 1
            elif isinstance(kv_data, (list, tuple)):
                num_layers = len(kv_data)
                for layer_data in kv_data:
                    if isinstance(layer_data, np.ndarray):
                        total_elements += layer_data.size
                        seq_length = max(seq_length, layer_data.shape[-1] if layer_data.ndim > 0 else 1)
                    elif isinstance(layer_data, (list, tuple)) and len(layer_data) > 0:
                        if isinstance(layer_data[0], (list, tuple)) and len(layer_data[0]) > 0:
                            total_elements += len(layer_data) * len(layer_data[0])
                            seq_length = max(seq_length, len(layer_data[0]))
                        else:
                            total_elements += len(layer_data)
                    else:
                        total_elements += 1
            else:
                total_elements = 1
            bytes_per_element = 4
            if isinstance(kv_data, dict):
                first_val = next(iter(kv_data.values()), None)
                if isinstance(first_val, np.ndarray):
                    bytes_per_element = first_val.dtype.itemsize
                elif isinstance(first_val, (list, tuple)) and len(first_val) > 0:
                    if isinstance(first_val[0], np.ndarray):
                        bytes_per_element = first_val[0].dtype.itemsize
                    elif isinstance(first_val[0], float):
                        bytes_per_element = 8
            elif isinstance(kv_data, np.ndarray):
                bytes_per_element = kv_data.dtype.itemsize
            size_mb = max(0.0, total_elements * bytes_per_element / (1024 * 1024))
            if size_mb > self.max_size_mb:
                logger.warning("Cache entry %.1fMB exceeds max cache size %.1fMB",
                             size_mb, self.max_size_mb)
            entry = CacheEntry(
                model_id=model_id,
                request_hash=req_hash,
                prompt_tokens=[],
                kv_tensors=kv_data,
                num_layers=num_layers,
                seq_length=seq_length,
                size_mb=size_mb,
            )
            if req_hash in self._cache:
                old = self._cache[req_hash]
                self._current_size_mb -= old.size_mb
            while self._current_size_mb + size_mb > self.max_size_mb and self._cache:
                self._evict_oldest()
            if size_mb > 0 and self._current_size_mb + size_mb > self.max_size_mb and not self._cache:
                logger.warning("Cache entry %.1fMB exceeds remaining space (max=%.1fMB, used=%.1fMB)",
                             size_mb, self.max_size_mb, self._current_size_mb)
            self._cache[req_hash] = entry
            self._current_size_mb += size_mb
            return entry

    def put_prefix_cache(self, model_id: str, prefix: str, kv_data: list[list[list[float]]]) -> CacheEntry:
        return self.put(model_id, prefix, kv_data, params={"type": "prefix"})

    def get_prefix_cache(self, model_id: str, prompt: str) -> CacheEntry | None:
        return self.get(model_id, prompt, params={"type": "prefix"})

    def _evict(self, key: str):
        entry = self._cache.pop(key, None)
        if entry:
            self._current_size_mb -= entry.size_mb
            self._evictions += 1

    def _evict_oldest(self):
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._current_size_mb -= entry.size_mb
            self._evictions += 1

    def create_partition(self, model_id: str, node_id: str, layer_start: int, layer_end: int, max_mb: float = 512.0) -> CachePartition:
        partition = CachePartition(
            node_id=node_id,
            model_id=model_id,
            layer_start=layer_start,
            layer_end=layer_end,
            max_size_mb=max_mb,
        )
        self._partitions.setdefault(model_id, {})[partition.partition_id] = partition
        return partition

    def get_partition(self, model_id: str, partition_id: str) -> CachePartition | None:
        return self._partitions.get(model_id, {}).get(partition_id)

    def get_model_partitions(self, model_id: str) -> list[CachePartition]:
        return list(self._partitions.get(model_id, {}).values())

    def get_distributed_cache_status(self, model_id: str) -> dict[str, Any]:
        partitions = self.get_model_partitions(model_id)
        return {
            "model_id": model_id,
            "num_partitions": len(partitions),
            "total_size_mb": sum(p.size_mb for p in partitions),
            "total_capacity_mb": sum(p.max_size_mb for p in partitions),
            "partitions": [
                {
                    "partition_id": p.partition_id,
                    "node_id": p.node_id,
                    "layers": f"{p.layer_start}-{p.layer_end}",
                    "usage_pct": round(p.usage_pct, 1),
                }
                for p in partitions
            ],
        }

    def clear(self):
        self._cache.clear()
        self._current_size_mb = 0.0

    def get_stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "entries": len(self._cache),
            "size_mb": round(self._current_size_mb, 2),
            "max_size_mb": self.max_size_mb,
            "usage_pct": round(self._current_size_mb / self.max_size_mb * 100, 1) if self.max_size_mb > 0 else 0,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
            "evictions": self._evictions,
            "partitions": sum(len(p) for p in self._partitions.values()),
        }


class DistributedKVCache:
    def __init__(self, local_cache: KVCacheManager, node_id: str = ""):
        self.local = local_cache
        self.node_id = node_id
        self.peer_caches: dict[str, dict[str, Any]] = {}

    def register_peer_cache(self, peer_id: str, status: dict[str, Any]):
        self.peer_caches[peer_id] = status

    def find_cached_request(self, model_id: str, prompt: str, params: dict[str, Any] | None = None) -> tuple[str, str | None]:
        local = self.local.get(model_id, prompt, params)
        if local:
            return self.node_id, local.cache_id
        for peer_id, status in self.peer_caches.items():
            if status.get("hit", False):
                return peer_id, status.get("cache_id")
        return "", None

    def compute_cache_affinity(self, model_id: str, prompt: str) -> str:
        h = int(hashlib.sha256(f"{model_id}:{prompt[:64]}".encode()).hexdigest(), 16)
        node_ids = [self.node_id] + [n for n in self.peer_caches.keys() if n]
        if not node_ids:
            return self.node_id or "local"
        return node_ids[h % len(node_ids)]

    def get_aggregate_stats(self) -> dict[str, Any]:
        local_stats = self.local.get_stats()
        peer_total_mb = sum(s.get("size_mb", 0) for s in self.peer_caches.values())
        peer_total_entries = sum(s.get("entries", 0) for s in self.peer_caches.values())
        return {
            "local": local_stats,
            "peer_count": len(self.peer_caches),
            "peer_total_mb": round(peer_total_mb, 2),
            "peer_total_entries": peer_total_entries,
            "aggregate_size_mb": round(local_stats["size_mb"] + peer_total_mb, 2),
            "aggregate_entries": local_stats["entries"] + peer_total_entries,
        }