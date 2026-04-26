"""GitHub integration - webhook receiver, commit-triggered training, model push."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from typing import Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GitHubConfig(BaseModel):
    repo_owner: str = ""
    repo_name: str = ""
    repo_url: str = ""
    branch: str = "main"
    access_token: str = ""
    webhook_secret: str = ""
    model_config_path: str = "netai.yaml"
    model_output_path: str = "models/"
    auto_train_on_push: bool = True
    auto_push_model: bool = True
    trigger_paths: list[str] = Field(default_factory=lambda: [
        "model/", "config/", "netai.yaml", "training/"
    ])


class CommitInfo(BaseModel):
    sha: str
    message: str = ""
    author: str = ""
    timestamp: str = ""
    added: list[str] = Field(default_factory=list)
    modified: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    url: str = ""


class WebhookEvent(BaseModel):
    event_type: str = "push"
    repo: str = ""
    branch: str = ""
    commits: list[CommitInfo] = Field(default_factory=list)
    sender: str = ""
    action: str = ""
    should_trigger: bool = False
    config_changed: bool = False
    timestamp: float = Field(default_factory=time.time)


class GitHubIntegration:
    def __init__(self, config: GitHubConfig | None = None):
        self.config = config or GitHubConfig()
        self._commit_handlers: list[Any] = []
        self._push_handlers: list[Any] = []
        self._last_sha: str = ""

    def on_commit(self, handler):
        self._commit_handlers.append(handler)

    def on_push(self, handler):
        self._push_handlers.append(handler)

    def verify_webhook(self, payload: bytes, signature: str) -> bool:
        if not self.config.webhook_secret:
            return False
        mac = hmac.new(
            self.config.webhook_secret.encode(),
            payload,
            hashlib.sha256,
        )
        expected = f"sha256={mac.hexdigest()}"
        return hmac.compare_digest(expected, signature)

    def parse_webhook_event(self, headers: dict[str, str], payload: dict[str, Any]) -> WebhookEvent:
        event_type = headers.get("X-GitHub-Event", "push")
        repo_info = payload.get("repository", {})
        repo_full = repo_info.get("full_name", self.config.repo_url)
        branch = payload.get("ref", "").replace("refs/heads/", "")
        sender = payload.get("sender", {}).get("login", "")
        event = WebhookEvent(
            event_type=event_type,
            repo=repo_full,
            branch=branch,
            sender=sender,
        )
        if event_type == "push":
            event.commits = [
                CommitInfo(
                    sha=c.get("id", ""),
                    message=c.get("message", ""),
                    author=c.get("author", {}).get("name", ""),
                    timestamp=c.get("timestamp", ""),
                    added=c.get("added", []),
                    modified=c.get("modified", []),
                    removed=c.get("removed", []),
                    url=c.get("url", ""),
                )
                for c in payload.get("commits", [])
            ]
            event.should_trigger = self._should_trigger(event)
            event.config_changed = self._config_changed(event)
        elif event_type == "pull_request":
            action = payload.get("action", "")
            event.action = action
            if action in ("closed", "merged"):
                pr = payload.get("pull_request", {})
                if pr.get("merged", False):
                    event.should_trigger = True
                    event.config_changed = True
        return event

    def _should_trigger(self, event: WebhookEvent) -> bool:
        if not self.config.auto_train_on_push:
            return False
        if event.branch != self.config.branch:
            return False
        all_changes = []
        for commit in event.commits:
            all_changes.extend(commit.added + commit.modified + commit.removed)
        for path in all_changes:
            for trigger in self.config.trigger_paths:
                if path.startswith(trigger):
                    return True
        return False

    def _config_changed(self, event: WebhookEvent) -> bool:
        all_changes = []
        for commit in event.commits:
            all_changes.extend(commit.added + commit.modified)
        return any(c == self.config.model_config_path or c.startswith("config/") for c in all_changes)

    async def process_webhook(self, event: WebhookEvent) -> dict[str, Any] | None:
        self._last_sha = event.commits[0].sha if event.commits else ""
        if not event.should_trigger:
            logger.info("Webhook event does not trigger training")
            return {"triggered": False, "reason": "no_trigger_paths"}
        for handler in self._commit_handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    result = await result
            except Exception as e:
                logger.error("Commit handler error: %s", e)
        for handler in self._push_handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    result = await result
            except Exception as e:
                logger.error("Push handler error: %s", e)
        return {
            "triggered": True,
            "job_config": self.config.model_config_path,
            "commits": len(event.commits),
            "config_changed": event.config_changed,
        }

    async def clone_or_pull(self, work_dir: str | None = None) -> str:
        if not work_dir:
            work_dir = tempfile.mkdtemp(prefix="netai-")
        branch = re.sub(r'[^\w./\-]', '', self.config.branch)
        repo_url = self.config.repo_url
        if self.config.access_token and "://" in repo_url:
            proto, rest = repo_url.split("://", 1)
            repo_url = f"{proto}://{self.config.access_token}@{rest}"
        if os.path.exists(os.path.join(work_dir, ".git")):
            subprocess.run(["git", "fetch", "--all"], cwd=work_dir, capture_output=True, timeout=60)
            subprocess.run(["git", "checkout", branch], cwd=work_dir, capture_output=True, timeout=30)
            subprocess.run(["git", "pull", "--rebase"], cwd=work_dir, capture_output=True, timeout=60)
        else:
            subprocess.run(
                ["git", "clone", "--branch", branch, repo_url, work_dir],
                capture_output=True, timeout=120,
            )
        return work_dir

    def read_config_from_repo(self, work_dir: str) -> dict[str, Any]:
        config_path = os.path.join(work_dir, self.config.model_config_path)
        if not os.path.exists(config_path):
            alt_paths = ["config.yaml", "training.yaml", "netai-config.yaml"]
            for alt in alt_paths:
                alt_path = os.path.join(work_dir, alt)
                if os.path.exists(alt_path):
                    config_path = alt_path
                    break
        if not os.path.exists(config_path):
            return {}
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    async def push_model(self, model_path: str, job_id: str, step: int, metrics: dict[str, float] | None = None):
        if not self.config.auto_push_model or not self.config.access_token:
            return
        work_dir = tempfile.mkdtemp(prefix="netai-model-push-")
        try:
            await self.clone_or_pull(work_dir)
            out_dir = os.path.join(work_dir, self.config.model_output_path, job_id)
            os.makedirs(out_dir, exist_ok=True)
            import shutil
            shutil.copy2(model_path, out_dir)
            meta = {
                "job_id": job_id,
                "step": step,
                "timestamp": time.time(),
                "metrics": metrics or {},
                "model_path": os.path.basename(model_path),
            }
            with open(os.path.join(out_dir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)
            subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True, timeout=30)
            subprocess.run(
                ["git", "commit", "-m", f"netai: add model checkpoint for {job_id} step {step}"],
                cwd=work_dir, capture_output=True, timeout=30,
            )
            subprocess.run(["git", "push"], cwd=work_dir, capture_output=True, timeout=60)
            logger.info("Model pushed to GitHub: %s step %d", job_id, step)
        except Exception as e:
            logger.error("Model push failed: %s", e)
        finally:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)

    def get_latest_commit(self) -> str:
        return self._last_sha

    def get_commit_diff(self, work_dir: str, sha: str) -> str:
        result = subprocess.run(
            ["git", "show", "--stat", sha],
            cwd=work_dir, capture_output=True, text=True, timeout=30,
        )
        return result.stdout