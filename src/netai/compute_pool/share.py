"""Compute share tracking — proof-of-compute and PPLNS reward distribution.

Inspired by mining pool share systems (P2Pool, PPLNS) but adapted for
distributed AI inference: shares prove compute was done, rewards are
free inference credits distributed proportionally to contribution.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import deque
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ShareStatus(str, Enum):
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    ORPHAN = "orphan"


class ShareDifficulty(str, Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"

    @staticmethod
    def from_vram_gb(vram_gb: float) -> "ShareDifficulty":
        if vram_gb <= 0:
            return ShareDifficulty.LIGHT
        if vram_gb < 4:
            return ShareDifficulty.LIGHT
        if vram_gb < 16:
            return ShareDifficulty.MEDIUM
        return ShareDifficulty.HEAVY

    def weight(self) -> float:
        return {
            ShareDifficulty.LIGHT: 1.0,
            ShareDifficulty.MEDIUM: 5.0,
            ShareDifficulty.HEAVY: 20.0,
        }.get(self, 1.0)

    def difficulty_target(self) -> int:
        return {
            ShareDifficulty.LIGHT: 2,
            ShareDifficulty.MEDIUM: 3,
            ShareDifficulty.HEAVY: 4,
        }.get(self, 3)


class ProofOfCompute(BaseModel):
    share_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    node_id: str = ""
    model_id: str = ""
    stage_index: int = 0
    input_hash: str = ""
    output_hash: str = ""
    compute_steps: int = 0
    vram_used_mb: float = 0.0
    latency_ms: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    nonce: int = 0

    def verify(self, difficulty_target: int = 3) -> bool:
        if difficulty_target < 1:
            difficulty_target = 1
        raw = json.dumps(
            {
                "node_id": self.node_id,
                "model_id": self.model_id,
                "stage_index": self.stage_index,
                "input_hash": self.input_hash,
                "output_hash": self.output_hash,
                "compute_steps": self.compute_steps,
                "nonce": self.nonce,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        h = hashlib.sha256(raw.encode()).hexdigest()
        prefix = h[:difficulty_target]
        try:
            val = int(prefix, 16)
        except ValueError:
            return False
        return val == 0


class ComputeShare(BaseModel):
    share_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    node_id: str = ""
    model_id: str = ""
    stage_index: int = 0
    difficulty: ShareDifficulty = ShareDifficulty.MEDIUM
    proof: ProofOfCompute | None = None
    status: ShareStatus = ShareStatus.PENDING
    weight: float = 1.0
    tokens_contributed: float = 0.0
    latency_ms: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    validated_at: float = 0.0


class ComputeContribution(BaseModel):
    node_id: str = ""
    total_shares: int = 0
    valid_shares: int = 0
    total_weight: float = 0.0
    total_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    _total_latency_ms: float = 0.0
    _latency_count: int = 0
    uptime_seconds: float = 0.0
    inference_credits: float = 0.0
    last_share_time: float = 0.0
    joined_at: float = Field(default_factory=time.time)

    @property
    def share_rate(self) -> float:
        if self.uptime_seconds <= 0:
            return 0.0
        return self.total_shares / max(self.uptime_seconds / 3600.0, 0.001)

    @property
    def reliability(self) -> float:
        if self.total_shares == 0:
            return 0.0
        return self.valid_shares / self.total_shares


class ShareLedger:
    def __init__(self, max_shares: int = 8640, share_window_hours: float = 72.0):
        self._shares: deque[ComputeShare] = deque(maxlen=max_shares)
        self._contributions: dict[str, ComputeContribution] = {}
        self._max_shares = max_shares
        self._window_seconds = share_window_hours * 3600

    def add_share(self, share: ComputeShare, skip_validation: bool = False) -> ComputeShare:
        if share.proof is not None and not skip_validation:
            target = share.difficulty.difficulty_target()
            if not share.proof.verify(target):
                share.status = ShareStatus.INVALID
                return share
        share.weight = share.difficulty.weight() * (1.0 / max(share.latency_ms / 1000.0, 0.1))
        share.tokens_contributed = share.weight * 100.0
        share.status = ShareStatus.VALID
        share.validated_at = time.time()
        self._shares.append(share)
        self._update_contribution(share)
        return share

    def validate_share(self, share: ComputeShare, difficulty_target: int = 3) -> bool:
        if share.proof is None:
            return False
        if difficulty_target < 1:
            difficulty_target = 1
        return share.proof.verify(difficulty_target)

    def _update_contribution(self, share: ComputeShare):
        c = self._contributions.setdefault(share.node_id, ComputeContribution(node_id=share.node_id))
        c.total_shares += 1
        if share.status == ShareStatus.VALID:
            c.valid_shares += 1
            c._total_latency_ms += share.latency_ms
            c._latency_count += 1
            c.avg_latency_ms = c._total_latency_ms / max(c._latency_count, 1)
        c.total_weight += share.weight
        c.total_tokens += share.tokens_contributed
        c.last_share_time = share.timestamp

    def prune_expired(self, now: float | None = None):
        now = now or time.time()
        cutoff = now - self._window_seconds
        while self._shares and self._shares[0].timestamp < cutoff:
            self._shares.popleft()

    def get_recent_shares(self, count: int = 100, node_id: str = "") -> list[ComputeShare]:
        shares = list(self._shares)
        if node_id:
            shares = [s for s in shares if s.node_id == node_id]
        return shares[-count:]

    def get_contribution(self, node_id: str) -> ComputeContribution | None:
        return self._contributions.get(node_id)

    def get_all_contributions(self) -> list[ComputeContribution]:
        return list(self._contributions.values())

    def get_leaderboard(self, limit: int = 50) -> list[ComputeContribution]:
        contribs = sorted(self._contributions.values(), key=lambda c: c.total_weight, reverse=True)
        return contribs[:limit]

    def total_share_count(self) -> int:
        return len(self._shares)


class PPLNSRewardCalculator:
    def __init__(self, window_n: int = 8640, credit_per_weight: float = 100.0):
        self._window_n = window_n
        self._credit_per_weight = credit_per_weight

    def calculate_rewards(self, ledger: ShareLedger, total_inference_budget: float = 0.0) -> dict[str, float]:
        recent = ledger.get_recent_shares(count=self._window_n)
        if not recent:
            return {}
        node_weights: dict[str, float] = {}
        for share in recent:
            if share.status != ShareStatus.VALID:
                continue
            node_weights[share.node_id] = node_weights.get(share.node_id, 0.0) + share.weight

        total_weight = sum(node_weights.values())
        if total_weight == 0:
            return {}

        rewards: dict[str, float] = {}
        for node_id, weight in node_weights.items():
            fraction = weight / total_weight
            credits = fraction * total_inference_budget if total_inference_budget > 0 else weight * self._credit_per_weight
            rewards[node_id] = credits

        return rewards

    def calculate_inference_priority(self, ledger: ShareLedger, node_id: str) -> float:
        c = ledger.get_contribution(node_id)
        if c is None or c.total_weight == 0:
            return 1.0
        all_weights = [c2.total_weight for c2 in ledger.get_all_contributions()]
        total = sum(all_weights)
        if total == 0:
            return 1.0
        percentile = sum(1 for w in all_weights if w <= c.total_weight) / len(all_weights)
        return max(0.01, percentile)