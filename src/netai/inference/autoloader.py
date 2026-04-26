"""Decentralized model catalog and auto-loading system.

ModelRegistry fetches the latest model catalog from GitHub (or uses a local
bundled copy). AutoLoader monitors the compute pool and automatically loads
models when enough VRAM is available, starting with mini models and scaling
up as more nodes jack in.

Size classes:
  - mini:  2M  to 100M  params — runs on any device
  - small: 100M to 350M params — runs on single consumer GPU
  - mid:   350M to 700M params — runs on high-VRAM GPU or pipeline-parallel
  - large: 700M+              params — requires distributed pipeline-parallel
"""

from __future__ import annotations

import json
import logging
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

CATALOG_URL = "https://raw.githubusercontent.com/netai-ai/models/main/catalog.json"
LOCAL_CATALOG_PATH = Path(__file__).parent.parent.parent / "models_catalog.json"
CATALOG_CACHE_PATH = Path.home() / ".config" / "netai" / "catalog_cache.json"
CATALOG_REFRESH_INTERVAL = 3600.0


class ModelSizeClass(str, Enum):
    MINI = "mini"
    SMALL = "small"
    MID = "mid"
    LARGE = "large"

    @classmethod
    def for_params(cls, params_m: float) -> "ModelSizeClass":
        if params_m <= 100:
            return cls.MINI
        elif params_m <= 350:
            return cls.SMALL
        elif params_m <= 700:
            return cls.MID
        else:
            return cls.LARGE

    @classmethod
    def min_vram_mb(cls, size_class: "ModelSizeClass") -> float:
        return {
            cls.MINI: 0,
            cls.SMALL: 4000,
            cls.MID: 17000,
            cls.LARGE: 40000,
        }[size_class]


class ModelEntry:
    __slots__ = (
        "model_id", "name", "architecture", "size_class", "params_m",
        "hidden_size", "num_layers", "num_heads", "vocab_size",
        "intermediate_size", "quantizations", "vram_required_mb",
        "context_length", "license", "huggingface_id", "description",
    )

    def __init__(self, data: dict[str, Any]):
        self.model_id = data["model_id"]
        self.name = data["name"]
        self.architecture = data["architecture"]
        sc = data.get("size_class", "")
        if not sc:
            sc = ModelSizeClass.for_params(data["params_m"]).value
        self.size_class = ModelSizeClass(sc)
        self.params_m = data["params_m"]
        self.hidden_size = data.get("hidden_size", 0)
        self.num_layers = data.get("num_layers", 0)
        self.num_heads = data.get("num_heads", 0)
        self.vocab_size = data.get("vocab_size", 0)
        self.intermediate_size = data.get("intermediate_size", 0)
        self.quantizations = data.get("quantizations", ["q4_k_m"])
        self.vram_required_mb = data.get("vram_required_mb", {})
        self.context_length = data.get("context_length", 4096)
        self.license = data.get("license", "")
        self.huggingface_id = data.get("huggingface_id", "")
        self.description = data.get("description", "")

    def vram_for_quant(self, quant: str = "q4_k_m") -> float:
        return self.vram_required_mb.get(quant, list(self.vram_required_mb.values())[0] if self.vram_required_mb else 0)

    def can_fit(self, available_vram_mb: float, quant: str = "q4_k_m") -> bool:
        required = self.vram_for_quant(quant)
        return 0 < required <= available_vram_mb * 0.85

    def to_config_dict(self, quant: str = "q4_k_m") -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.name,
            "num_shards": 1,
            "quantization": quant,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "vocab_size": self.vocab_size,
            "intermediate_size": self.intermediate_size,
            "params_m": self.params_m,
            "context_length": self.context_length,
        }


class ModelRegistry:
    def __init__(self, catalog_path: str | Path | None = None):
        self._catalog_path = Path(catalog_path) if catalog_path else LOCAL_CATALOG_PATH
        self._models: dict[str, ModelEntry] = {}
        self._last_fetch: float = 0
        self._fetch_interval: float = CATALOG_REFRESH_INTERVAL

    def load_local(self) -> bool:
        path = self._catalog_path
        if not path.exists():
            cache_path = CATALOG_CACHE_PATH
            if cache_path.exists():
                path = cache_path
            else:
                logger.warning("No model catalog found at %s or %s", self._catalog_path, cache_path)
                return False
        try:
            with open(path) as f:
                data = json.load(f)
            self._parse_catalog(data)
            logger.info("Loaded %d models from catalog %s", len(self._models), path)
            return True
        except Exception as e:
            logger.error("Failed to load model catalog: %s", e)
            return False

    async def fetch_remote(self, url: str = CATALOG_URL) -> bool:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.warning("Catalog fetch failed: HTTP %d", resp.status)
                        return False
                    data = await resp.json(content_type=None)
        except Exception as e:
            logger.warning("Catalog fetch error: %s", e)
            return False
        self._parse_catalog(data)
        self._last_fetch = time.time()
        try:
            CATALOG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CATALOG_CACHE_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
        logger.info("Fetched and cached %d models from remote catalog", len(self._models))
        return True

    async def refresh(self) -> bool:
        if time.time() - self._last_fetch < self._fetch_interval:
            return True
        remote_ok = await self.fetch_remote()
        if not remote_ok:
            if not self._models:
                return self.load_local()
            return True
        return True

    def _parse_catalog(self, data: dict[str, Any]):
        models_list = data.get("models", [])
        self._models = {}
        for m in models_list:
            try:
                entry = ModelEntry(m)
                self._models[entry.model_id] = entry
            except Exception as e:
                logger.warning("Skipping invalid model entry: %s", e)

    def get(self, model_id: str) -> ModelEntry | None:
        return self._models.get(model_id)

    def list_models(self, size_class: ModelSizeClass | None = None) -> list[ModelEntry]:
        models = list(self._models.values())
        if size_class:
            models = [m for m in models if m.size_class == size_class]
        return sorted(models, key=lambda m: m.params_m)

    def models_for_vram(self, available_vram_mb: float, quant: str = "q4_k_m") -> list[ModelEntry]:
        return sorted(
            [m for m in self._models.values() if m.can_fit(available_vram_mb, quant)],
            key=lambda m: m.params_m,
        )

    def voting_priority(self, votes: dict[str, float] | None = None) -> list[ModelEntry]:
        models = list(self._models.values())
        if votes:
            models.sort(key=lambda m: votes.get(m.model_id, 0), reverse=True)
        else:
            models.sort(key=lambda m: (0 if m.size_class == ModelSizeClass.MINI else 1, m.params_m))
        return models

    @property
    def size_classes(self) -> dict[str, dict[str, Any]]:
        return {
            ModelSizeClass.MINI.value: {"min_params_m": 2, "max_params_m": 100},
            ModelSizeClass.SMALL.value: {"min_params_m": 100, "max_params_m": 350},
            ModelSizeClass.MID.value: {"min_params_m": 350, "max_params_m": 700},
            ModelSizeClass.LARGE.value: {"min_params_m": 700, "max_params_m": None},
        }


class AutoLoader:
    def __init__(
        self,
        registry: ModelRegistry,
        available_vram_mb: float = 0,
        available_nodes: int = 1,
        min_nodes_for_pipeline: int = 2,
        preferred_quant: str = "q4_k_m",
        max_concurrent_models: int = 3,
    ):
        self.registry = registry
        self.available_vram_mb = available_vram_mb
        self.available_nodes = available_nodes
        self.min_nodes_for_pipeline = min_nodes_for_pipeline
        self.preferred_quant = preferred_quant
        self.max_concurrent_models = max_concurrent_models
        self._loaded_models: dict[str, dict[str, Any]] = {}
        self._pending_loads: set[str] = set()
        self._load_history: list[tuple[str, float, float]] = []

    def update_resources(self, available_vram_mb: float, available_nodes: int = 1):
        self.available_vram_mb = available_vram_mb
        self.available_nodes = available_nodes

    def compute_load_plan(
        self,
        votes: dict[str, float] | None = None,
        force_models: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        plan: list[dict[str, Any]] = []
        remaining_vram = self.available_vram_mb * 0.80

        if force_models:
            for model_id in force_models:
                entry = self.registry.get(model_id)
                if entry and entry.can_fit(remaining_vram, self.preferred_quant):
                    vram = entry.vram_for_quant(self.preferred_quant)
                    plan.append({
                        "model_id": entry.model_id,
                        "name": entry.name,
                        "size_class": entry.size_class.value,
                        "params_m": entry.params_m,
                        "vram_mb": vram,
                        "quant": self.preferred_quant,
                        "shards": max(1, int(vram / (remaining_vram / max(self.available_nodes, 1)))),
                        "priority": len(plan) + 1,
                        "config": entry.to_config_dict(self.preferred_quant),
                    })
                    remaining_vram -= vram
                if len(plan) >= self.max_concurrent_models:
                    break
            return plan

        prioritized = self.registry.voting_priority(votes)
        for entry in prioritized:
            if entry.model_id in self._loaded_models or entry.model_id in self._pending_loads:
                continue
            if not entry.can_fit(remaining_vram, self.preferred_quant):
                continue
            vram = entry.vram_for_quant(self.preferred_quant)
            can_pipeline = self.available_nodes >= self.min_nodes_for_pipeline
            if entry.size_class in (ModelSizeClass.MID, ModelSizeClass.LARGE) and not can_pipeline:
                if self.available_nodes < 1:
                    continue
            plan.append({
                "model_id": entry.model_id,
                "name": entry.name,
                "size_class": entry.size_class.value,
                "params_m": entry.params_m,
                "vram_mb": vram,
                "quant": self.preferred_quant,
                "shards": max(1, int(vram / max(remaining_vram, 1) * self.available_nodes)),
                "priority": len(plan) + 1,
                "config": entry.to_config_dict(self.preferred_quant),
            })
            remaining_vram -= vram
            if len(plan) >= self.max_concurrent_models:
                break

        if not plan:
            mini_models = self.registry.list_models(ModelSizeClass.MINI)
            for entry in mini_models:
                if entry.can_fit(remaining_vram, self.preferred_quant):
                    vram = entry.vram_for_quant(self.preferred_quant)
                    plan.append({
                        "model_id": entry.model_id,
                        "name": entry.name,
                        "size_class": entry.size_class.value,
                        "params_m": entry.params_m,
                        "vram_mb": vram,
                        "quant": self.preferred_quant,
                        "shards": 1,
                        "priority": len(plan) + 1,
                        "config": entry.to_config_dict(self.preferred_quant),
                    })
                    remaining_vram -= vram
                    if len(plan) >= 1:
                        break

        return plan

    def mark_loaded(self, model_id: str, vram_mb: float):
        self._loaded_models[model_id] = {
            "loaded_at": time.time(),
            "vram_mb": vram_mb,
        }
        self._load_history.append((model_id, time.time(), vram_mb))
        self._pending_loads.discard(model_id)

    def mark_unloaded(self, model_id: str):
        self._loaded_models.pop(model_id, None)

    def mark_pending(self, model_id: str):
        self._pending_loads.add(model_id)

    def get_loaded_models(self) -> list[dict[str, Any]]:
        result = []
        for model_id, info in self._loaded_models.items():
            entry = self.registry.get(model_id)
            if entry:
                result.append({
                    "model_id": model_id,
                    "name": entry.name,
                    "size_class": entry.size_class.value,
                    "params_m": entry.params_m,
                    "vram_mb": info["vram_mb"],
                    "loaded_at": info["loaded_at"],
                })
        return sorted(result, key=lambda x: x["params_m"])

    def should_unload(self, model_id: str) -> bool:
        entry = self.registry.get(model_id)
        if not entry:
            return False
        info = self._loaded_models.get(model_id)
        if not info:
            return False
        loaded_vram = info["vram_mb"]
        total_loaded_vram = sum(m["vram_mb"] for m in self.get_loaded_models())
        if total_loaded_vram > self.available_vram_mb * 0.90:
            models = self.get_loaded_models()
            if models and models[0]["model_id"] == model_id:
                return True
        return False

    def get_status(self) -> dict[str, Any]:
        loaded = self.get_loaded_models()
        plan = self.compute_load_plan()
        return {
            "available_vram_mb": self.available_vram_mb,
            "available_nodes": self.available_nodes,
            "loaded_models": loaded,
            "pending_loads": list(self._pending_loads),
            "recommended_loads": plan,
            "preferred_quant": self.preferred_quant,
            "max_concurrent_models": self.max_concurrent_models,
            "catalog_size": len(self.registry._models),
        }