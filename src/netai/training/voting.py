"""Voting system for model selection, resource pledging, and governance."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VoteType(str, Enum):
    MODEL_SELECT = "model_select"
    RESOURCE_PLEDGE = "resource_pledge"
    TRAIN_START = "train_start"
    TRAIN_STOP = "train_stop"
    CONFIG_CHANGE = "config_change"
    GROUP_CREATE = "group_create"
    GROUP_MEMBER_ADD = "group_member_add"
    GROUP_MEMBER_REMOVE = "group_member_remove"
    MODEL_MERGE = "model_merge"
    CHECKPOINT_APPROVE = "checkpoint_approve"


class VoteWeight(str, Enum):
    EQUAL = "equal"
    BY_RESOURCE = "by_resource"
    BY_STAKE = "by_stake"
    BY_REPUTATION = "by_reputation"


class ProposalStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PASSED = "passed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXECUTED = "executed"


class ResourcePledge(BaseModel):
    user_id: str
    node_id: str = ""
    cpu_cores: int = 0
    gpu_count: int = 0
    ram_gb: float = 0.0
    gpu_vram_mb: list[int] = Field(default_factory=list)
    time_hours: float = 24.0
    auto_extend: bool = False
    max_jobs: int = 3
    priority: int = 0
    group_id: str = ""
    timestamp: float = Field(default_factory=time.time)
    signature: str = ""

    @property
    def compute_score(self) -> float:
        score = self.cpu_cores * 1.0
        score += self.gpu_count * 10.0
        score += min(self.ram_gb, 128.0) * 0.5
        for vram in self.gpu_vram_mb:
            score += vram / 1024.0 * 2.0
        score *= min(self.time_hours, 168.0) / 24.0
        return score

    @property
    def summary(self) -> str:
        parts = []
        if self.cpu_cores:
            parts.append(f"{self.cpu_cores} CPU cores")
        if self.gpu_count:
            parts.append(f"{self.gpu_count} GPU(s)")
        if self.ram_gb:
            parts.append(f"{self.ram_gb:.0f}GB RAM")
        if self.time_hours:
            parts.append(f"{self.time_hours:.0f}h")
        return ", ".join(parts) if parts else "No resources"


class Proposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    description: str = ""
    vote_type: VoteType = VoteType.MODEL_SELECT
    proposer_id: str = ""
    created_at: float = Field(default_factory=time.time)
    voting_deadline: float = 0.0
    status: ProposalStatus = ProposalStatus.DRAFT
    config: dict[str, Any] = Field(default_factory=dict)
    votes_for: int = 0
    votes_against: int = 0
    vote_weight_type: VoteWeight = VoteWeight.BY_RESOURCE
    weighted_for: float = 0.0
    weighted_against: float = 0.0
    voters: dict[str, str] = Field(default_factory=dict)
    quorum: float = 0.5
    threshold: float = 0.6
    group_id: str = ""
    resource_pledges: list[ResourcePledge] = Field(default_factory=list)
    execution_result: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)

    @property
    def total_votes(self) -> int:
        return self.votes_for + self.votes_against

    @property
    def total_weighted(self) -> float:
        return self.weighted_for + self.weighted_against

    @property
    def is_expired(self) -> bool:
        if self.voting_deadline <= 0:
            return False
        return time.time() > self.voting_deadline

    @property
    def result(self) -> ProposalStatus:
        if self.status in (ProposalStatus.CANCELLED, ProposalStatus.EXECUTED):
            return self.status
        if not self.is_expired and self.status == ProposalStatus.ACTIVE:
            return ProposalStatus.ACTIVE
        total_w = self.weighted_for + self.weighted_against
        if total_w == 0:
            return ProposalStatus.FAILED
        if self.quorum > 0 and total_w < self.quorum:
            return ProposalStatus.FAILED
        ratio = self.weighted_for / total_w if total_w > 0 else 0
        if ratio >= self.threshold:
            return ProposalStatus.PASSED
        return ProposalStatus.FAILED


class Vote(BaseModel):
    vote_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    proposal_id: str = ""
    voter_id: str = ""
    node_id: str = ""
    choice: str = ""
    weight: float = 1.0
    resource_pledge: ResourcePledge | None = None
    signature: str = ""
    timestamp: float = Field(default_factory=time.time)
    reasoning: str = ""


class UserModelProposal(BaseModel):
    model_name: str
    architecture: str = "transformer"
    description: str = ""
    repo_url: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    estimated_resources: dict[str, float] = Field(default_factory=dict)
    proposer_id: str = ""
    tags: list[str] = Field(default_factory=list)


class VotingEngine:
    def __init__(self, vote_weight_type: VoteWeight = VoteWeight.BY_RESOURCE):
        self.proposals: dict[str, Proposal] = {}
        self.votes: dict[str, list[Vote]] = {}
        self.pledges: dict[str, ResourcePledge] = {}
        self.model_proposals: dict[str, UserModelProposal] = {}
        self.reputation: dict[str, float] = {}
        self.vote_weight_type = vote_weight_type
        self._handlers: dict[VoteType, Any] = {}
        self._default_voting_period = 86400 * 3  # 3 days

    def on_proposal_executed(self, vote_type: VoteType, handler):
        self._handlers[vote_type] = handler

    def create_model_proposal(
        self,
        model: UserModelProposal,
        proposer_id: str,
        group_id: str = "",
        voting_deadline: float | None = None,
        config: dict[str, Any] | None = None,
    ) -> Proposal:
        proposal = Proposal(
            title=f"Train model: {model.model_name}",
            description=model.description,
            vote_type=VoteType.MODEL_SELECT,
            proposer_id=proposer_id,
            voting_deadline=voting_deadline or (time.time() + self._default_voting_period),
            status=ProposalStatus.ACTIVE,
            config=config or {},
            group_id=group_id,
            tags=model.tags,
        )
        proposal.config["model"] = model.model_dump()
        self.proposals[proposal.proposal_id] = proposal
        self.votes[proposal.proposal_id] = []
        self.model_proposals[proposal.proposal_id] = model
        logger.info("Model proposal created: %s (%s)", proposal.title, proposal.proposal_id)
        return proposal

    def create_resource_pledge(
        self,
        pledge: ResourcePledge,
    ) -> Proposal:
        proposal = Proposal(
            title=f"Resource pledge: {pledge.summary}",
            description=f"{pledge.user_id} pledges {pledge.summary}",
            vote_type=VoteType.RESOURCE_PLEDGE,
            proposer_id=pledge.user_id,
            status=ProposalStatus.ACTIVE,
            voting_deadline=time.time() + 3600,
            group_id=pledge.group_id,
        )
        proposal.resource_pledges.append(pledge)
        proposal.config["pledge"] = pledge.model_dump()
        self.proposals[proposal.proposal_id] = proposal
        self.votes[proposal.proposal_id] = []
        self.pledges[pledge.user_id] = pledge
        return proposal

    def create_train_proposal(
        self,
        job_config: dict[str, Any],
        proposer_id: str,
        group_id: str = "",
        voting_deadline: float | None = None,
    ) -> Proposal:
        proposal = Proposal(
            title=f"Start training: {job_config.get('model_name', 'custom')}",
            description=f"Proposed by {proposer_id}",
            vote_type=VoteType.TRAIN_START,
            proposer_id=proposer_id,
            voting_deadline=voting_deadline or (time.time() + 86400),
            status=ProposalStatus.ACTIVE,
            config=job_config,
            group_id=group_id,
        )
        self.proposals[proposal.proposal_id] = proposal
        self.votes[proposal.proposal_id] = []
        return proposal

    def cast_vote(
        self,
        proposal_id: str,
        voter_id: str,
        choice: str,
        weight: float | None = None,
        node_id: str = "",
        reasoning: str = "",
        pledge: ResourcePledge | None = None,
    ) -> Vote | None:
        if proposal_id not in self.proposals:
            logger.error("Proposal %s not found", proposal_id)
            return None
        proposal = self.proposals[proposal_id]
        if proposal.status != ProposalStatus.ACTIVE:
            logger.error("Proposal %s is not active (status=%s)", proposal_id, proposal.status)
            return None
        if proposal.is_expired:
            logger.error("Proposal %s voting deadline has passed", proposal_id)
            return None
        if voter_id in proposal.voters:
            old = proposal.voters[voter_id]
            old_vote = self._find_vote(proposal_id, voter_id)
            old_weight = old_vote.weight if old_vote else 0.0
            if old == "for":
                proposal.votes_for -= 1
                proposal.weighted_for -= old_weight
            elif old == "against":
                proposal.votes_against -= 1
                proposal.weighted_against -= old_weight
        computed_weight = weight if weight is not None else self._compute_weight(voter_id, pledge)
        vote = Vote(
            proposal_id=proposal_id,
            voter_id=voter_id,
            node_id=node_id,
            choice=choice,
            weight=computed_weight,
            resource_pledge=pledge,
            reasoning=reasoning,
        )
        if choice == "for":
            proposal.votes_for += 1
            proposal.weighted_for += computed_weight
        elif choice == "against":
            proposal.votes_against += 1
            proposal.weighted_against += computed_weight
        proposal.voters[voter_id] = choice
        self.votes[proposal_id].append(vote)
        logger.info("Vote cast: %s on %s by %s (weight=%.2f)", choice, proposal_id, voter_id, computed_weight)
        self._check_proposal(proposal_id)
        return vote

    def _compute_weight(self, voter_id: str, pledge: ResourcePledge | None = None) -> float:
        if self.vote_weight_type == VoteWeight.EQUAL:
            return 1.0
        elif self.vote_weight_type == VoteWeight.BY_RESOURCE:
            p = pledge or self.pledges.get(voter_id)
            if p:
                return p.compute_score
            return 0.1
        elif self.vote_weight_type == VoteWeight.BY_STAKE:
            p = self.pledges.get(voter_id)
            if p:
                return p.compute_score
            return 0.1
        elif self.vote_weight_type == VoteWeight.BY_REPUTATION:
            return self.reputation.get(voter_id, 1.0)
        return 1.0

    def _find_vote(self, proposal_id: str, voter_id: str):
        for vote in self.votes.get(proposal_id, []):
            if vote.voter_id == voter_id:
                return vote
        return None

    def _check_proposal(self, proposal_id: str):
        proposal = self.proposals.get(proposal_id)
        if not proposal or proposal.status != ProposalStatus.ACTIVE:
            return
        result = proposal.result
        if result in (ProposalStatus.PASSED, ProposalStatus.FAILED):
            proposal.status = result
            logger.info("Proposal %s: %s (for=%.2f, against=%.2f)", proposal_id, result, proposal.weighted_for, proposal.weighted_against)
            if result == ProposalStatus.PASSED:
                self._execute_proposal(proposal_id)

    def _execute_proposal(self, proposal_id: str):
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return
        handler = self._handlers.get(proposal.vote_type)
        if handler:
            try:
                result = handler(proposal)
                proposal.execution_result = result if isinstance(result, dict) else {"status": str(result)}
            except Exception as e:
                proposal.execution_result = {"error": str(e)}
                logger.error("Proposal execution failed: %s", e)
        proposal.status = ProposalStatus.EXECUTED

    def get_proposal(self, proposal_id: str) -> Proposal | None:
        return self.proposals.get(proposal_id)

    def list_proposals(
        self,
        status: ProposalStatus | None = None,
        vote_type: VoteType | None = None,
        group_id: str | None = None,
    ) -> list[Proposal]:
        results = list(self.proposals.values())
        if status:
            results = [p for p in results if p.status == status]
        if vote_type:
            results = [p for p in results if p.vote_type == vote_type]
        if group_id:
            results = [p for p in results if p.group_id == group_id]
        return results

    def cancel_proposal(self, proposal_id: str, canceller_id: str) -> bool:
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return False
        if proposal.proposer_id != canceller_id:
            return False
        proposal.status = ProposalStatus.CANCELLED
        return True

    def update_reputation(self, user_id: str, delta: float):
        current = self.reputation.get(user_id, 1.0)
        self.reputation[user_id] = max(0.1, current + delta)

    def get_leaderboard(self) -> list[dict[str, Any]]:
        entries = []
        for uid, pledge in self.pledges.items():
            entries.append({
                "user_id": uid,
                "score": pledge.compute_score,
                "cpu": pledge.cpu_cores,
                "gpu": pledge.gpu_count,
                "ram_gb": pledge.ram_gb,
                "reputation": self.reputation.get(uid, 1.0),
                "summary": pledge.summary,
            })
        entries.sort(key=lambda x: x["score"], reverse=True)
        for i, e in enumerate(entries):
            e["rank"] = i + 1
        return entries

    def get_cluster_resources(self, group_id: str = "") -> dict[str, Any]:
        pledges = [p for p in self.pledges.values() if not group_id or p.group_id == group_id]
        total_cpu = sum(p.cpu_cores for p in pledges)
        total_gpu = sum(p.gpu_count for p in pledges)
        total_ram = sum(p.ram_gb for p in pledges)
        total_vram = sum(v for p in pledges for v in p.gpu_vram_mb)
        return {
            "total_cpu_cores": total_cpu,
            "total_gpu_count": total_gpu,
            "total_ram_gb": total_ram,
            "total_vram_mb": total_vram,
            "num_contributors": len(pledges),
            "pledges": [p.model_dump() for p in pledges],
        }