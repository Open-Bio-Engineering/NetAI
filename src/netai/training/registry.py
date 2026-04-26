"""Model registry and version management."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    DRAFT = "draft"
    TRAINING = "training"
    EVALUATING = "evaluating"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelVersion(BaseModel):
    version_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_id: str = ""
    version: str = "1.0.0"
    step: int = 0
    loss: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    checkpoint_path: str = ""
    model_hash: str = ""
    size_mb: float = 0.0
    created_at: float = Field(default_factory=time.time)
    created_by: str = ""
    is_best: bool = False
    is_published: bool = False
    tags: list[str] = Field(default_factory=list)
    parent_version: str = ""
    commit_sha: str = ""


class ModelEntry(BaseModel):
    model_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    name: str = ""
    architecture: str = "transformer"
    description: str = ""
    group_id: str = ""
    owner_id: str = ""
    status: ModelStatus = ModelStatus.DRAFT
    versions: list[ModelVersion] = Field(default_factory=list)
    best_version: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    total_training_hours: float = 0.0
    total_steps: int = 0
    license: str = "Apache-2.0"
    repo_url: str = ""
    dataset: str = ""


class ModelRegistry:
    def __init__(self, storage_dir: str = "./model_registry"):
        self.storage_dir = storage_dir
        self.models: dict[str, ModelEntry] = {}
        self._version_index: dict[str, dict[str, ModelVersion]] = {}
        os.makedirs(storage_dir, exist_ok=True)

    def register_model(
        self,
        name: str,
        architecture: str = "transformer",
        description: str = "",
        owner_id: str = "",
        group_id: str = "",
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        license: str = "Apache-2.0",
        repo_url: str = "",
    ) -> ModelEntry:
        model = ModelEntry(
            name=name,
            architecture=architecture,
            description=description,
            owner_id=owner_id,
            group_id=group_id,
            config=config or {},
            tags=tags or [],
            license=license,
            repo_url=repo_url,
        )
        self.models[model.model_id] = model
        self._version_index[model.model_id] = {}
        logger.info("Model registered: %s (%s)", name, model.model_id)
        return model

    def add_version(
        self,
        model_id: str,
        version: str,
        step: int,
        loss: float,
        checkpoint_path: str = "",
        metrics: dict[str, float] | None = None,
        created_by: str = "",
        parent_version: str = "",
        commit_sha: str = "",
        tags: list[str] | None = None,
    ) -> ModelVersion | None:
        model = self.models.get(model_id)
        if not model:
            return None
        mv = ModelVersion(
            model_id=model_id,
            version=version,
            step=step,
            loss=loss,
            checkpoint_path=checkpoint_path,
            metrics=metrics or {},
            created_by=created_by,
            parent_version=parent_version,
            commit_sha=commit_sha,
            tags=tags or [],
        )
        if checkpoint_path and os.path.exists(checkpoint_path):
            mv.size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            with open(checkpoint_path, "rb") as f:
                data = f.read()
                mv.model_hash = hashlib.sha256(data).hexdigest()[:16]
        model.versions.append(mv)
        model.updated_at = time.time()
        model.total_steps = max(model.total_steps, step)
        self._version_index[model_id][version] = mv
        if not model.best_version or loss < self._get_best_loss(model_id):
            model.best_version = version
            mv.is_best = True
            for v in model.versions:
                if v.version != version:
                    v.is_best = False
        logger.info("Version %s added to model %s (loss=%.4f)", version, model.name, loss)
        return mv

    def _get_best_loss(self, model_id: str) -> float:
        model = self.models.get(model_id)
        if not model or not model.versions:
            return float("inf")
        return min(v.loss for v in model.versions)

    def get_version(self, model_id: str, version: str) -> ModelVersion | None:
        return self._version_index.get(model_id, {}).get(version)

    def get_best_version(self, model_id: str) -> ModelVersion | None:
        model = self.models.get(model_id)
        if not model or not model.best_version:
            return None
        return self.get_version(model_id, model.best_version)

    def publish_version(self, model_id: str, version: str) -> bool:
        mv = self.get_version(model_id, version)
        if not mv:
            return False
        mv.is_published = True
        model = self.models.get(model_id)
        if model:
            model.status = ModelStatus.PUBLISHED
            model.updated_at = time.time()
        return True

    def deprecate_version(self, model_id: str, version: str) -> bool:
        mv = self.get_version(model_id, version)
        if not mv:
            return False
        mv.is_published = False
        return True

    def list_models(
        self,
        group_id: str | None = None,
        owner_id: str | None = None,
        status: ModelStatus | None = None,
        architecture: str | None = None,
    ) -> list[ModelEntry]:
        results = list(self.models.values())
        if group_id:
            results = [m for m in results if m.group_id == group_id]
        if owner_id:
            results = [m for m in results if m.owner_id == owner_id]
        if status:
            results = [m for m in results if m.status == status]
        if architecture:
            results = [m for m in results if m.architecture == architecture]
        return results

    def get_model(self, model_id: str) -> ModelEntry | None:
        return self.models.get(model_id)

    def search(self, query: str) -> list[ModelEntry]:
        q = query.lower()
        results = []
        for m in self.models.values():
            if (q in m.name.lower() or q in m.description.lower()
                    or q in m.architecture.lower() or any(q in t.lower() for t in m.tags)):
                results.append(m)
        return results

    def delete_model(self, model_id: str) -> bool:
        model = self.models.pop(model_id, None)
        if not model:
            return False
        self._version_index.pop(model_id, None)
        model_dir = os.path.join(self.storage_dir, model_id)
        if os.path.exists(model_dir):
            import shutil
            shutil.rmtree(model_dir, ignore_errors=True)
        return True

    def get_model_stats(self, model_id: str) -> dict[str, Any]:
        model = self.models.get(model_id)
        if not model:
            return {}
        versions = model.versions
        return {
            "model_id": model_id,
            "name": model.name,
            "status": model.status.value,
            "total_versions": len(versions),
            "best_version": model.best_version,
            "best_loss": self._get_best_loss(model_id),
            "total_steps": model.total_steps,
            "total_training_hours": model.total_training_hours,
            "published_versions": sum(1 for v in versions if v.is_published),
        }

    def export_manifest(self, model_id: str) -> dict[str, Any]:
        model = self.models.get(model_id)
        if not model:
            return {}
        return {
            "model_id": model.model_id,
            "name": model.name,
            "architecture": model.architecture,
            "license": model.license,
            "versions": [
                {
                    "version": v.version,
                    "loss": v.loss,
                    "step": v.step,
                    "is_published": v.is_published,
                    "is_best": v.is_best,
                    "created_at": v.created_at,
                    "created_by": v.created_by,
                    "size_mb": v.size_mb,
                    "model_hash": v.model_hash,
                    "commit_sha": v.commit_sha,
                }
                for v in model.versions
            ],
        }