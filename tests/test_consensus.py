"""Tests for consensus-based pipeline coordination."""

import asyncio
import time
import pytest

from netai.consensus.pipeline_coordinator import (
    PipelineCoordinator,
    PipelineProposal,
)


class TestPipelineProposal:
    def test_proposal_creation_defaults(self):
        p = PipelineProposal(model_id="test-model", proposer_node_id="node-1")
        assert p.model_id == "test-model"
        assert p.proposer_node_id == "node-1"
        assert p.votes_for == 0
        assert p.votes_against == 0
        assert p.total_nodes == 0
        assert len(p.stage_assignments) == 0
        assert p.accepted is None

    def test_proposal_is_not_accepted_without_votes(self):
        p = PipelineProposal(model_id="m", proposer_node_id="n1", total_nodes=5)
        assert p.is_accepted is False
        assert p.is_rejected is False

    def test_majority_threshold_calculation(self):
        for total, expected in [(1, 1), (2, 2), (3, 2), (4, 3), (5, 3), (6, 4), (7, 4), (100, 51)]:
            p = PipelineProposal(model_id="m", proposer_node_id="n1", total_nodes=total)
            assert p.majority_threshold == expected, f"Failed for total={total}"

    def test_proposal_acceptance_with_majority(self):
        p = PipelineProposal(model_id="m", proposer_node_id="n1", total_nodes=5)
        p.votes_for = 3
        assert p.is_accepted is True

    def test_proposal_rejection_with_majority_against(self):
        p = PipelineProposal(model_id="m", proposer_node_id="n1", total_nodes=5)
        p.votes_against = 3
        assert p.is_rejected is True

    def test_proposal_expiration(self):
        p = PipelineProposal(
            model_id="m", proposer_node_id="n1",
            expiration=time.time() - 10, total_nodes=5,
        )
        assert p.is_expired is True
        assert p.is_accepted is False
        assert p.is_rejected is True

    def test_proposal_impossible_to_win_is_rejected(self):
        p = PipelineProposal(model_id="m", proposer_node_id="n1", total_nodes=5)
        p.votes_for = 1
        p.votes_against = 3
        assert p.is_rejected is True


class TestPipelineCoordinator:
    @pytest.fixture
    def coordinator(self):
        return PipelineCoordinator(node_id="coordinator-1")

    def test_register_and_remove_nodes(self, coordinator):
        coordinator.register_node("n1")
        coordinator.register_node("n2")
        assert coordinator.total_nodes == 2
        coordinator.remove_node("n1")
        assert coordinator.total_nodes == 1

    def test_elect_coordinator_single_node(self):
        coordinator = PipelineCoordinator(node_id="node-aaa")
        leader = asyncio.run(coordinator.elect_coordinator())
        assert leader == "node-aaa"

    def test_elect_coordinator_highest_id_wins(self):
        coordinator = PipelineCoordinator(node_id="node-bbb")
        coordinator.register_node("node-ccc")
        coordinator.register_node("node-aaa")
        leader = asyncio.run(coordinator.elect_coordinator())
        assert leader == "node-ccc"

    def test_vote_and_accept_proposal(self, coordinator):
        coordinator.register_node("n2")
        coordinator.register_node("n3")

        proposal = PipelineProposal(
            model_id="gpt2", proposer_node_id="n1",
            stage_assignments={"0": "n1", "1": "n2", "2": "n3"},
            total_nodes=coordinator.total_nodes,
        )
        proposal.votes_for = 1
        proposal.voted_nodes.add("n1")
        coordinator.inject_proposal(proposal)

        asyncio.run(coordinator.vote_on_proposal(proposal.proposal_id, True, "n2"))
        assert proposal.votes_for == 2
        assert proposal.is_accepted is True

        active = coordinator.get_active_pipelines()
        assert "gpt2" in active
        assert active["gpt2"]["model_id"] == "gpt2"

    def test_vote_against_rejects_proposal(self, coordinator):
        coordinator.register_node("n2")
        coordinator.register_node("n3")

        proposal = PipelineProposal(
            model_id="llama", proposer_node_id="n1",
            stage_assignments={"0": "n1", "1": "n2", "2": "n3"},
            total_nodes=coordinator.total_nodes,
        )
        proposal.votes_for = 1
        proposal.voted_nodes.add("n1")
        coordinator.inject_proposal(proposal)

        asyncio.run(coordinator.vote_on_proposal(proposal.proposal_id, False, "n2"))
        asyncio.run(coordinator.vote_on_proposal(proposal.proposal_id, False, "n3"))
        assert proposal.votes_against == 2
        assert proposal.is_rejected is True
        assert proposal.is_accepted is False

    def test_no_double_voting(self, coordinator):
        coordinator.register_node("n2")

        proposal = PipelineProposal(
            model_id="m", proposer_node_id="n1",
            total_nodes=coordinator.total_nodes,
        )
        proposal.votes_for = 1
        proposal.voted_nodes.add("n1")
        coordinator.inject_proposal(proposal)

        asyncio.run(coordinator.vote_on_proposal(proposal.proposal_id, True, "n2"))
        second = asyncio.run(coordinator.vote_on_proposal(proposal.proposal_id, False, "n2"))
        assert proposal.votes_for == 2
        assert proposal.votes_against == 0

    def test_reassign_stages_on_node_failure(self, coordinator):
        coordinator.register_node("n1")
        coordinator.register_node("n2")
        coordinator.register_node("n3")

        proposal = PipelineProposal(
            model_id="gpt2", proposer_node_id="n1",
            stage_assignments={"0": "n1", "1": "n2", "2": "n3", "3": "n2"},
            total_nodes=coordinator.total_nodes,
        )
        proposal.votes_for = 2
        proposal.accepted = True
        coordinator.inject_proposal(proposal)

        coordinator._active_pipelines["gpt2"] = {
            "proposal_id": proposal.proposal_id,
            "model_id": "gpt2",
            "stage_assignments": dict(proposal.stage_assignments),
            "accepted_at": time.time(),
        }

        new_assignments = asyncio.run(coordinator.reassign_stages("gpt2", "n2"))
        assert new_assignments is not None
        for si, nid in new_assignments.items():
            assert nid != "n2"

    def test_reassign_nonexistent_pipeline(self, coordinator):
        result = asyncio.run(coordinator.reassign_stages("nonexistent", "n-fail"))
        assert result is None

    def test_concurrent_proposals(self, coordinator):
        coordinator.register_node("n1")
        coordinator.register_node("n2")

        p1 = PipelineProposal(
            proposal_id="p1", model_id="gpt2", proposer_node_id="n1",
            stage_assignments={"0": "n1", "1": "n2"},
            total_nodes=coordinator.total_nodes,
        )
        p2 = PipelineProposal(
            proposal_id="p2", model_id="gpt2", proposer_node_id="n2",
            stage_assignments={"0": "n2", "1": "n1"},
            total_nodes=coordinator.total_nodes,
        )
        coordinator.inject_proposal(p1)
        coordinator.inject_proposal(p2)

        pending = coordinator.get_pending_proposals()
        assert len(pending) == 2

    def test_cleanup_expired_proposals(self, coordinator):
        p = PipelineProposal(
            model_id="m", proposer_node_id="n1",
            expiration=time.time() - 5, total_nodes=coordinator.total_nodes,
        )
        coordinator.inject_proposal(p)
        removed = coordinator.cleanup_expired()
        assert p.proposal_id in removed
        assert coordinator.get_proposal(p.proposal_id) is None

    def test_get_status(self, coordinator):
        coordinator.register_node("n1")
        coordinator.register_node("n2")

        asyncio.run(coordinator.elect_coordinator())

        status = coordinator.get_status()
        assert status["node_id"] == "coordinator-1"
        assert status["total_nodes"] == 2
        assert len(status["known_nodes"]) == 2
        assert "n1" in status["known_nodes"]
        assert "n2" in status["known_nodes"]

    def test_active_pipelines_empty_initially(self, coordinator):
        assert coordinator.get_active_pipelines() == {}

    def test_get_proposal_unknown_returns_none(self, coordinator):
        assert coordinator.get_proposal("bogus-id") is None

    def test_vote_unknown_proposal_returns_none(self, coordinator):
        result = asyncio.run(coordinator.vote_on_proposal("bogus-id", True, "n1"))
        assert result is None

    def test_tie_handling_no_majority(self, coordinator):
        coordinator.register_node("n1")
        coordinator.register_node("n2")
        coordinator.register_node("n3")
        coordinator.register_node("n4")

        proposal = PipelineProposal(
            model_id="m", proposer_node_id="n1",
            total_nodes=coordinator.total_nodes,
        )
        proposal.votes_for = 1
        proposal.voted_nodes.add("n1")
        coordinator.inject_proposal(proposal)

        asyncio.run(coordinator.vote_on_proposal(proposal.proposal_id, True, "n2"))
        asyncio.run(coordinator.vote_on_proposal(proposal.proposal_id, False, "n3"))

        assert proposal.votes_for == 2
        assert proposal.votes_against == 1
        assert proposal.is_accepted is False
        assert proposal.is_rejected is False

    def test_reassign_with_only_self_node(self, coordinator):
        proposal = PipelineProposal(
            model_id="m", proposer_node_id="n-fail",
            stage_assignments={"0": "n-fail", "1": "n-fail"},
            total_nodes=1,
        )
        proposal.accepted = True
        coordinator.inject_proposal(proposal)
        coordinator._active_pipelines["m"] = {
            "proposal_id": proposal.proposal_id,
            "model_id": "m",
            "stage_assignments": dict(proposal.stage_assignments),
            "accepted_at": time.time(),
        }

        new_assignments = asyncio.run(coordinator.reassign_stages("m", "n-fail"))
        assert new_assignments is not None
        assert all(v == "coordinator-1" for v in new_assignments.values())


def run_tests():
    return True
