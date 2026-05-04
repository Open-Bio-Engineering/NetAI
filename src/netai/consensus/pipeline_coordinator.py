"""Consensus protocol for distributed pipeline stage assignment.

Nodes collaboratively agree on how to assign transformer layers across
the peer-to-peer pool. Uses majority voting for proposal acceptance, simple
leader election (highest node_id wins), and handles node failures via
stage reassignment.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from netai.inference.pipeline_executor import PipelineExecutor, PipelineStage
from netai.p2p.handshake import HandshakeProtocol

logger = logging.getLogger(__name__)


class PipelineProposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_id: str = ""
    proposer_node_id: str = ""
    stage_assignments: dict[str, str] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    expiration: float = Field(default_factory=lambda: time.time() + 30.0)
    votes_for: int = 0
    votes_against: int = 0
    voted_nodes: set[str] = Field(default_factory=set, exclude=True)
    accepted: bool | None = None
    total_nodes: int = Field(default=0, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expiration

    @property
    def majority_threshold(self) -> int:
        return max(self.total_nodes // 2 + 1, 1)

    @property
    def is_accepted(self) -> bool:
        if self.accepted is not None:
            return self.accepted
        if self.is_expired:
            return False
        if self.total_nodes <= 0:
            return False
        return self.votes_for >= self.majority_threshold

    @property
    def is_rejected(self) -> bool:
        if self.accepted is False:
            return True
        if self.is_expired:
            return True
        if self.total_nodes <= 0:
            return False
        remaining = self.total_nodes - (self.votes_for + self.votes_against)
        threshold = self.majority_threshold
        if self.votes_against >= threshold:
            return True
        if self.votes_for + remaining < threshold:
            return True
        return False


class PipelineCoordinator:
    """Coordinates consensus-driven pipeline stage assignments across nodes.

    Uses majority voting (N/2 + 1) for proposal acceptance, simple leader
    election where the highest node_id wins, and handles node failures
    by reassigning stages to remaining nodes.
    """

    def __init__(self, node_id: str = ""):
        self.node_id: str = node_id or uuid.uuid4().hex[:12]
        self._proposals: dict[str, PipelineProposal] = {}
        self._active_pipelines: dict[str, dict] = {}
        self._pending_proposals: list[PipelineProposal] = []
        self._known_nodes: dict[str, dict[str, Any]] = {}
        self._leader_node_id: str | None = None
        self._coordinator_lock = asyncio.Lock()
        self._total_nodes_cache: int = 1
        self._total_nodes_cache_time: float = 0.0

    def register_node(self, node_id: str, capabilities: dict[str, Any] | None = None) -> None:
        self._known_nodes[node_id] = capabilities or {}
        self._invalidate_cache()

    def remove_node(self, node_id: str) -> None:
        self._known_nodes.pop(node_id, None)
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        self._total_nodes_cache_time = 0.0

    @property
    def total_nodes(self) -> int:
        if time.time() - self._total_nodes_cache_time < 1.0:
            return self._total_nodes_cache
        count = max(len(self._known_nodes), 1)
        self._total_nodes_cache = count
        self._total_nodes_cache_time = time.time()
        return count

    async def propose_pipeline(
        self,
        model_id: str,
        config: Any,
        node_capabilities: list[dict[str, Any]],
        timeout_s: float = 30.0,
    ) -> PipelineProposal:
        """Propose a pipeline layout for peer voting.

        Uses the PipelineExecutor to plan stage assignments, wraps them in
        a PipelineProposal, and registers it for voting.
        """
        async with self._coordinator_lock:
            plan = PipelineExecutor().plan_pipeline(model_id, config, node_capabilities)

            stage_assignments: dict[str, str] = {}
            for s in plan:
                stage_assignments[str(s.stage_index)] = s.node_id

            proposal = PipelineProposal(
                model_id=model_id,
                proposer_node_id=self.node_id,
                stage_assignments=stage_assignments,
                expiration=time.time() + timeout_s,
                total_nodes=self.total_nodes,
            )
            proposal.votes_for = 1
            proposal.voted_nodes.add(self.node_id)

            self._proposals[proposal.proposal_id] = proposal
            logger.info(
                "Proposed pipeline %s for %s: %d stages across %d nodes",
                proposal.proposal_id, model_id, len(stage_assignments), self.total_nodes,
            )
            return proposal

    async def vote_on_proposal(self, proposal_id: str, vote: bool, node_id: str) -> PipelineProposal | None:
        async with self._coordinator_lock:
            proposal = self._proposals.get(proposal_id)
            if proposal is None:
                logger.warning("Vote on unknown proposal %s", proposal_id)
                return None

            if node_id in proposal.voted_nodes:
                logger.debug("Node %s already voted on proposal %s", node_id, proposal_id)
                return proposal

            proposal.voted_nodes.add(node_id)
            if vote:
                proposal.votes_for += 1
            else:
                proposal.votes_against += 1

            logger.info(
                "Node %s voted %s on proposal %s (%d for, %d against, threshold=%d)",
                node_id, "for" if vote else "against", proposal_id,
                proposal.votes_for, proposal.votes_against, proposal.majority_threshold,
            )

            if proposal.is_accepted:
                proposal.accepted = True
                self._active_pipelines[proposal.model_id] = {
                    "proposal_id": proposal.proposal_id,
                    "model_id": proposal.model_id,
                    "stage_assignments": proposal.stage_assignments,
                    "accepted_at": time.time(),
                }
                logger.info("Proposal %s ACCEPTED for model %s", proposal_id, proposal.model_id)
            elif proposal.is_rejected:
                proposal.accepted = False
                logger.info("Proposal %s REJECTED for model %s", proposal_id, proposal.model_id)

            return proposal

    async def elect_coordinator(self) -> str:
        node_ids = list(self._known_nodes.keys())
        if not node_ids:
            elected = self.node_id
        else:
            elected = max([self.node_id] + node_ids)

        self._leader_node_id = elected
        logger.info("Leader elected: %s (out of %d nodes)", elected, self.total_nodes)
        return elected

    async def reassign_stages(self, model_id: str, failed_node_id: str) -> dict[str, str] | None:
        async with self._coordinator_lock:
            pipeline = self._active_pipelines.get(model_id)
            if not pipeline:
                logger.warning("No active pipeline for %s to reassign", model_id)
                return None

            assignments: dict[str, str] = dict(pipeline["stage_assignments"])
            affected_stages = [
                si for si, nid in assignments.items() if nid == failed_node_id
            ]

            if not affected_stages:
                logger.info("No stages on failed node %s for model %s", failed_node_id, model_id)
                return assignments

            alive_nodes = [
                nid for nid in self._known_nodes
                if nid != failed_node_id
            ]
            if not alive_nodes:
                alive_nodes = [self.node_id]

            for i, stage_index in enumerate(sorted(affected_stages, key=int)):
                new_node = alive_nodes[i % len(alive_nodes)]
                assignments[stage_index] = new_node
                logger.info(
                    "Reassigned stage %s of %s from %s → %s",
                    stage_index, model_id, failed_node_id, new_node,
                )

            pipeline["stage_assignments"] = assignments
            pipeline["reassigned_at"] = time.time()
            return assignments

    def get_active_pipelines(self) -> dict[str, dict]:
        return dict(self._active_pipelines)

    def get_proposal(self, proposal_id: str) -> PipelineProposal | None:
        return self._proposals.get(proposal_id)

    def get_pending_proposals(self) -> list[PipelineProposal]:
        return [p for p in self._proposals.values() if p.accepted is None and not p.is_expired]

    def cleanup_expired(self) -> list[str]:
        removed = []
        for pid, proposal in list(self._proposals.items()):
            if proposal.is_expired:
                self._proposals.pop(pid, None)
                removed.append(pid)
                logger.debug("Cleaned up expired proposal %s", pid)
        return removed

    def get_status(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "leader": self._leader_node_id,
            "total_nodes": self.total_nodes,
            "known_nodes": list(self._known_nodes.keys()),
            "active_pipelines": list(self._active_pipelines.keys()),
            "pending_proposals": len(self.get_pending_proposals()),
            "total_proposals": len(self._proposals),
        }

    def inject_proposal(self, proposal: PipelineProposal) -> None:
        """Inject a proposal directly (for testing concurrent proposals)."""
        proposal.total_nodes = self.total_nodes
        self._proposals[proposal.proposal_id] = proposal
