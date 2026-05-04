"""Model cache management — lists, prunes, verifies, and searches cached models.

Integrates with ModelDownloader for cache directory access and HF URL resolution.
Tracks LRU usage via cache_hit() for intelligent pruning.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from pydantic import BaseModel, Field

from netai.inference.downloader import ModelDownloader, DEFAULT_CACHE_DIR, HF_DOWNLOAD_BASE

logger = logging.getLogger(__name__)

_CACHE_METADATA_FILE = "netai_cache_metadata.json"


class CacheHitRequest(BaseModel):
    model_id: str = Field(..., description="Model ID in repo format (e.g. org/modelname)")

    model_config = {"json_schema_extra": {"examples": [{"model_id": "gpt2"}]}}


class CacheStatsResponse(BaseModel):
    total_size_bytes: int
    total_size_mb: float
    model_count: int
    disk_free_bytes: int
    disk_free_gb: float
    cache_dir: str
    hit_rate: float = 0.0
    total_hits: int = 0
    timestamp: float = Field(default_factory=time.time)


@dataclass
class CachedModelInfo:
    model_id: str
    cache_dir: str
    format_type: str = "safetensors"
    architecture: str = "unknown"
    param_count: int = 0
    file_count: int = 0
    total_size_bytes: int = 0
    downloaded_at: float = 0.0
    last_accessed_at: float = 0.0
    access_count: int = 0
    verified: bool = False
    revision: str = "main"
    files: dict[str, int] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)

    @property
    def downloaded_date(self) -> str:
        if self.downloaded_at <= 0:
            return "unknown"
        return datetime.datetime.fromtimestamp(self.downloaded_at).isoformat()

    @property
    def last_accessed_date(self) -> str:
        if self.last_accessed_at <= 0:
            return "never"
        return datetime.datetime.fromtimestamp(self.last_accessed_at).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "format_type": self.format_type,
            "architecture": self.architecture,
            "param_count": self.param_count,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "size_mb": round(self.size_mb, 2),
            "downloaded_at": self.downloaded_at,
            "downloaded_date": self.downloaded_date,
            "last_accessed_at": self.last_accessed_at,
            "last_accessed_date": self.last_accessed_date,
            "access_count": self.access_count,
            "verified": self.verified,
            "revision": self.revision,
            "files": self.files,
        }


class ModelCacheManager:
    """Manages locally cached model files with LRU tracking, integrity checks, and search.

    Wraps the existing ModelDownloader's cache directory. Maintains a metadata
    file tracking access patterns, sizes, and download dates.
    """

    def __init__(self, cache_dir: str = "", downloader: ModelDownloader | None = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._downloader = downloader
        os.makedirs(self.cache_dir, exist_ok=True)
        self._metadata_path = os.path.join(self.cache_dir, _CACHE_METADATA_FILE)
        self._metadata: dict[str, dict[str, Any]] = {}
        self._total_hits: int = 0
        self._total_requests: int = 0
        self._load_metadata()

    @property
    def downloader(self) -> ModelDownloader:
        if self._downloader is None:
            self._downloader = ModelDownloader(cache_dir=self.cache_dir)
        return self._downloader

    def _load_metadata(self) -> None:
        if os.path.exists(self._metadata_path):
            try:
                with open(self._metadata_path) as f:
                    data = json.load(f)
                self._metadata = data.get("models", {})
                self._total_hits = data.get("total_hits", 0)
                self._total_requests = data.get("total_requests", 0)
            except Exception:
                logger.warning("Failed to load cache metadata, starting fresh")
                self._metadata = {}

    def _save_metadata(self) -> None:
        try:
            with open(self._metadata_path, "w") as f:
                json.dump(
                    {
                        "models": self._metadata,
                        "total_hits": self._total_hits,
                        "total_requests": self._total_requests,
                        "updated_at": time.time(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.warning("Failed to save cache metadata: %s", e)

    def _model_dir(self, model_id: str) -> str:
        return os.path.join(self.cache_dir, model_id.replace("/", "--"))

    def _detect_format(self, model_dir: str) -> str:
        for fname in os.listdir(model_dir):
            if fname.endswith(".safetensors"):
                return "safetensors"
            if fname.endswith(".gguf"):
                return "gguf"
            if fname.endswith(".onnx"):
                return "onnx"
        return "bin"

    def _read_config(self, model_dir: str) -> dict[str, Any]:
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _estimate_param_count(self, model_dir: str, config: dict[str, Any]) -> int:
        total_bytes = 0
        for fname in os.listdir(model_dir):
            fpath = os.path.join(model_dir, fname)
            if os.path.isfile(fpath) and any(
                fname.endswith(ext) for ext in (".safetensors", ".bin", ".pt", ".gguf")
            ):
                total_bytes += os.path.getsize(fpath)
        if total_bytes > 0:
            return total_bytes // 2
        hidden = config.get("hidden_size", 0)
        layers = config.get("num_hidden_layers", config.get("num_layers", 0))
        if hidden and layers:
            return (12 * hidden * hidden * layers) // 2
        return 0

    def _build_info(self, model_id: str, model_dir: str) -> CachedModelInfo | None:
        if not os.path.isdir(model_dir):
            return None
        config = self._read_config(model_dir)
        files: dict[str, int] = {}
        total_bytes = 0
        for fname in os.listdir(model_dir):
            fpath = os.path.join(model_dir, fname)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                files[fname] = size
                total_bytes += size

        meta = self._metadata.get(model_id, {})

        return CachedModelInfo(
            model_id=model_id,
            cache_dir=self.cache_dir,
            format_type=self._detect_format(model_dir),
            architecture=config.get("architectures", [config.get("model_type", "unknown")])[0]
            if config
            else meta.get("architecture", "unknown"),
            param_count=self._estimate_param_count(model_dir, config),
            file_count=len(files),
            total_size_bytes=total_bytes,
            downloaded_at=meta.get("downloaded_at", 0.0),
            last_accessed_at=meta.get("last_accessed_at", 0.0),
            access_count=meta.get("access_count", 0),
            verified=meta.get("verified", False),
            revision=meta.get("revision", "main"),
            files=files,
            config=config,
        )

    def list_models(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if not os.path.isdir(self.cache_dir):
            return results
        seen: set[str] = set()
        for name in sorted(os.listdir(self.cache_dir)):
            model_dir = os.path.join(self.cache_dir, name)
            if not os.path.isdir(model_dir):
                continue
            if name in (_CACHE_METADATA_FILE,):
                continue
            model_id = name.replace("--", "/")
            if model_id in seen:
                continue
            seen.add(model_id)
            info = self._build_info(model_id, model_dir)
            if info is not None:
                results.append(info.to_dict())
        return results

    def get_model_info(self, model_id: str) -> dict[str, Any] | None:
        model_dir = self._model_dir(model_id)
        info = self._build_info(model_id, model_dir)
        if info is None:
            return None
        return info.to_dict()

    def delete_model(self, model_id: str) -> dict[str, Any]:
        model_dir = self._model_dir(model_id)
        if not os.path.isdir(model_dir):
            return {"status": "not_found", "model_id": model_id}

        total_bytes = 0
        file_count = 0
        for fname in os.listdir(model_dir):
            fpath = os.path.join(model_dir, fname)
            if os.path.isfile(fpath):
                total_bytes += os.path.getsize(fpath)
                file_count += 1

        shutil.rmtree(model_dir)
        self._metadata.pop(model_id, None)
        self._save_metadata()
        logger.info("Deleted cached model: %s (%.1f MB, %d files)", model_id, total_bytes / (1024 * 1024), file_count)
        return {
            "status": "deleted",
            "model_id": model_id,
            "freed_bytes": total_bytes,
            "freed_mb": round(total_bytes / (1024 * 1024), 2),
            "files_removed": file_count,
        }

    def prune_cache(self, keep_latest_n: int = 5) -> dict[str, Any]:
        models = self.list_models()
        if len(models) <= keep_latest_n:
            return {
                "status": "no_prune_needed",
                "total_models": len(models),
                "kept": len(models),
                "removed": 0,
            }

        models.sort(key=lambda m: m.get("last_accessed_at", 0), reverse=True)
        to_remove = models[keep_latest_n:]
        removed: list[dict[str, Any]] = []
        total_freed = 0
        for m in to_remove:
            result = self.delete_model(m["model_id"])
            removed.append(result)
            total_freed += result.get("freed_bytes", 0)

        return {
            "status": "pruned",
            "total_models_before": len(models),
            "kept": keep_latest_n,
            "removed_count": len(removed),
            "removed": [r["model_id"] for r in removed],
            "freed_bytes": total_freed,
            "freed_mb": round(total_freed / (1024 * 1024), 2),
        }

    def cache_stats(self) -> CacheStatsResponse:
        models = self.list_models()
        total_bytes = sum(m.get("total_size_bytes", 0) for m in models)
        try:
            stat = os.statvfs(self.cache_dir)
            disk_free = stat.f_frsize * stat.f_bavail
        except Exception:
            disk_free = 0
        hit_rate = self._total_hits / max(self._total_requests, 1)
        return CacheStatsResponse(
            total_size_bytes=total_bytes,
            total_size_mb=round(total_bytes / (1024 * 1024), 2),
            model_count=len(models),
            disk_free_bytes=disk_free,
            disk_free_gb=round(disk_free / (1024 ** 3), 2),
            cache_dir=self.cache_dir,
            hit_rate=round(hit_rate, 4),
            total_hits=self._total_hits,
        )

    def verify_integrity(self, model_id: str) -> dict[str, Any]:
        model_dir = self._model_dir(model_id)
        if not os.path.isdir(model_dir):
            return {"status": "not_found", "model_id": model_id}

        results: list[dict[str, Any]] = []
        all_ok = True
        for fname in sorted(os.listdir(model_dir)):
            fpath = os.path.join(model_dir, fname)
            if not os.path.isfile(fpath):
                continue
            sha256 = hashlib.sha256()
            try:
                with open(fpath, "rb") as f:
                    while chunk := f.read(8 * 1024 * 1024):
                        sha256.update(chunk)
                digest = sha256.hexdigest()
                results.append({
                    "filename": fname,
                    "size_bytes": os.path.getsize(fpath),
                    "sha256": digest,
                    "valid": True,
                })
            except Exception as e:
                results.append({
                    "filename": fname,
                    "size_bytes": os.path.getsize(fpath) if os.path.exists(fpath) else 0,
                    "sha256": "",
                    "valid": False,
                    "error": str(e),
                })
                all_ok = False

        if all_ok:
            if model_id in self._metadata:
                self._metadata[model_id]["verified"] = True
                self._save_metadata()

        return {
            "status": "ok" if all_ok else "corrupt",
            "model_id": model_id,
            "all_valid": all_ok,
            "files_checked": len(results),
            "files": results,
        }

    def get_download_urls(self, model_id: str) -> dict[str, Any]:
        model_dir = self._model_dir(model_id)
        if not os.path.isdir(model_dir):
            return {"status": "not_found", "model_id": model_id}

        revision = self._metadata.get(model_id, {}).get("revision", "main")
        urls: dict[str, str] = {}
        for fname in sorted(os.listdir(model_dir)):
            fpath = os.path.join(model_dir, fname)
            if os.path.isfile(fpath) and not fname.startswith("."):
                original = fname.replace("_", "/") if "/" not in fname else fname
                urls[fname] = f"{HF_DOWNLOAD_BASE}/{model_id}/resolve/{revision}/{original}"

        return {
            "status": "ok",
            "model_id": model_id,
            "revision": revision,
            "huggingface_url": f"{HF_DOWNLOAD_BASE}/{model_id}",
            "files": urls,
        }

    def search_models(self, query: str) -> list[dict[str, Any]]:
        q = query.lower().strip()
        if not q:
            return self.list_models()

        results: list[dict[str, Any]] = []
        for m in self.list_models():
            model_id = m.get("model_id", "").lower()
            arch = m.get("architecture", "").lower()
            fmt = m.get("format_type", "").lower()
            if q in model_id or q in arch or q in fmt or model_id == q:
                results.append(m)
        return results

    def cache_hit(self, model_id: str) -> dict[str, Any]:
        self._total_requests += 1

        model_dir = self._model_dir(model_id)
        exists = os.path.isdir(model_dir)

        if exists:
            self._total_hits += 1
            now = time.time()
            if model_id in self._metadata:
                self._metadata[model_id]["last_accessed_at"] = now
                self._metadata[model_id]["access_count"] = self._metadata[model_id].get("access_count", 0) + 1
            else:
                self._metadata[model_id] = {
                    "last_accessed_at": now,
                    "access_count": 1,
                    "downloaded_at": now,
                    "verified": False,
                    "revision": "main",
                }
            self._save_metadata()
            logger.debug("Cache hit: %s (access #%d)", model_id, self._metadata[model_id]["access_count"])
        else:
            logger.debug("Cache miss: %s", model_id)

        return {
            "model_id": model_id,
            "hit": exists,
            "access_count": self._metadata.get(model_id, {}).get("access_count", 0),
            "last_accessed_at": self._metadata.get(model_id, {}).get("last_accessed_at", 0),
        }

    def export_model_info(self) -> dict[str, Any]:
        models = self.list_models()
        stats = self.cache_stats()
        return {
            "exported_at": time.time(),
            "exported_date": datetime.datetime.now().isoformat(),
            "cache_dir": self.cache_dir,
            "summary": stats.model_dump(),
            "models": models,
        }
