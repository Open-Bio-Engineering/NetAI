"""Edge-case tests for P2P networking, coordinator, federation, and inference routing."""

import asyncio
import time
import pytest
import numpy as np

from netai.p2p.network import PeerInfo, PeerTable, P2PNode, PeerMessage
from netai.training.coordinator import DistributedTrainingCoordinator
from netai.training.engine import (
    TrainingConfig, LocalTrainer, GradientSyncServer, GradientCompressor,
)
from netai.training.federation import Federation, FederationNode
from netai.inference.router import (
    InferenceNode, InferenceLoadBalancer, RoutingStrategy,
)
from netai.inference.engine import InferenceStatus, InferenceRequest


class TestP2PAsyncMethods:
    """Test P2P PeerTable async methods."""

    @pytest.mark.asyncio
    async def test_add_peer(self):
        table = PeerTable(self_id="self")
        result = await table.add_peer(PeerInfo(node_id="n1", host="1.2.3.4", port=8000))
        assert result is True
        assert "n1" in table._peers

    @pytest.mark.asyncio
    async def test_add_self_rejected(self):
        table = PeerTable(self_id="self")
        result = await table.add_peer(PeerInfo(node_id="self", host="1.1.1.1", port=8000))
        assert result is False

    @pytest.mark.asyncio
    async def test_update_reliability_success(self):
        table = PeerTable(self_id="self")
        await table.add_peer(PeerInfo(node_id="n1", host="1.2.3.4", port=8000))
        await table.update_reliability("n1", True)
        record = table._peers["n1"]
        assert record.reliability_score > 1.0

    @pytest.mark.asyncio
    async def test_update_reliability_failure(self):
        table = PeerTable(self_id="self")
        await table.add_peer(PeerInfo(node_id="n1", host="1.2.3.4", port=8000))
        await table.update_reliability("n1", False)
        record = table._peers["n1"]
        assert record.reliability_score < 1.0

    @pytest.mark.asyncio
    async def test_update_reliability_nonexistent_peer_ignored(self):
        table = PeerTable(self_id="self")
        await table.update_reliability("ghost", True)
        assert "ghost" not in table._peers

    @pytest.mark.asyncio
    async def test_update_reliability_repeated_failures(self):
        table = PeerTable(self_id="self")
        await table.add_peer(PeerInfo(node_id="n1", host="1.2.3.4", port=8000))
        for _ in range(10):
            await table.update_reliability("n1", False)
        record = table._peers["n1"]
        assert record.reliability_score < 1.0

    @pytest.mark.asyncio
    async def test_eviction_when_at_max(self):
        table = PeerTable(self_id="self", max_peers=3)
        for i in range(5):
            await table.add_peer(PeerInfo(node_id=f"p{i}", host=f"h{i}", port=i))
        all_p = await table.get_all_peers()
        assert len(all_p) <= 3

    @pytest.mark.asyncio
    async def test_no_eviction_when_under_max(self):
        table = PeerTable(self_id="self", max_peers=5)
        await table.add_peer(PeerInfo(node_id="n1", host="1.1.1.1", port=8000))
        await table.add_peer(PeerInfo(node_id="n2", host="2.2.2.2", port=8000))
        assert len(table._peers) == 2

    @pytest.mark.asyncio
    async def test_stale_peers_reliability_degrades(self):
        table = PeerTable(self_id="self")
        await table.add_peer(PeerInfo(node_id="n1", host="1.1.1.1", port=8000, last_heartbeat=0))
        initial_score = table._peers["n1"].reliability_score
        await table.get_alive_peers()
        assert table._peers["n1"].reliability_score < initial_score

    @pytest.mark.asyncio
    async def test_remove_peer(self):
        table = PeerTable(self_id="self")
        await table.add_peer(PeerInfo(node_id="n1", host="1.1.1.1", port=8000))
        result = await table.remove_peer("n1")
        assert result is True
        assert "n1" not in table._peers


class TestPeerMessageSigning:
    """Test P2P message signing."""

    def test_sign_message_with_identity(self):
        node = P2PNode(port=0, node_id="sign-test")
        node._node_identity = None
        msg = PeerMessage(sender_id="sign-test", msg_type="test", data={"key": "value"})
        signed = node._sign_message(msg)
        assert isinstance(signed, dict)
        assert "signature" in signed
        assert signed["signature"] == ""

    def test_unsigned_message_empty_signature(self):
        msg = PeerMessage(sender_id="anon", msg_type="test", data={})
        assert msg.signature == ""


class TestCoordinatorEdgeCases:
    """Test DistributedTrainingCoordinator edge cases."""

    @pytest.mark.asyncio
    async def test_stop_training_nonexistent_job(self):
        p2p = P2PNode(port=0)
        coord = DistributedTrainingCoordinator(p2p)
        result = await coord.stop_training("nonexistent-job")
        assert result is None

    @pytest.mark.asyncio
    async def test_submit_job_creates_job(self):
        p2p = P2PNode(port=0)
        coord = DistributedTrainingCoordinator(p2p)
        config = TrainingConfig(model_name="test", total_steps=10)
        job = await coord.submit_job(config)
        assert job is not None
        assert hasattr(job, "job_id")
        assert len(job.job_id) > 0

    @pytest.mark.asyncio
    async def test_list_jobs(self):
        p2p = P2PNode(port=0)
        coord = DistributedTrainingCoordinator(p2p)
        config = TrainingConfig(model_name="test", total_steps=10)
        job = await coord.submit_job(config)
        jobs = coord.list_jobs()
        assert len(jobs) >= 1
        assert any(j["job_id"] == job.job_id for j in jobs)

    @pytest.mark.asyncio
    async def test_get_job_status(self):
        p2p = P2PNode(port=0)
        coord = DistributedTrainingCoordinator(p2p)
        config = TrainingConfig(model_name="test", total_steps=10)
        job = await coord.submit_job(config)
        status = coord.get_job_status(job.job_id)
        assert status is not None
        assert status["job_id"] == job.job_id

    @pytest.mark.asyncio
    async def test_get_job_status_nonexistent(self):
        p2p = P2PNode(port=0)
        coord = DistributedTrainingCoordinator(p2p)
        status = coord.get_job_status("nonexistent")
        assert status is None


class TestFederationEdgeCases:
    """Test Federation module edge cases."""

    @pytest.mark.asyncio
    async def test_register_peer(self):
        fed = Federation(cluster_id="fed-node-1", cluster_endpoint="http://localhost:18001")
        node = FederationNode(node_id="peer-1", name="Peer 1", endpoint="http://peer1:8000", resources={"cpu_cores": 8, "gpu_count": 2})
        result = await fed.register_peer(node)
        assert result is True
        assert "peer-1" in fed.peers

    @pytest.mark.asyncio
    async def test_register_self_rejected(self):
        fed = Federation(cluster_id="fed-node-1", cluster_endpoint="http://localhost:18001")
        node = FederationNode(node_id="fed-node-1", name="Self", endpoint="http://self:8000", resources={})
        result = await fed.register_peer(node)
        assert result is False

    @pytest.mark.asyncio
    async def test_unregister_peer(self):
        fed = Federation(cluster_id="fed-node-1", cluster_endpoint="http://localhost:18001")
        node = FederationNode(node_id="peer-1", name="Peer 1", endpoint="http://peer1:8000", resources={"cpu_cores": 8})
        await fed.register_peer(node)
        await fed.unregister_peer("peer-1")
        assert "peer-1" not in fed.peers

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_peer(self):
        fed = Federation(cluster_id="fed-node-1", cluster_endpoint="http://localhost:18001")
        await fed.unregister_peer("ghost")

    @pytest.mark.asyncio
    async def test_propose_shared_training(self):
        fed = Federation(cluster_id="fed-node-1", cluster_endpoint="http://localhost:18001")
        result = await fed.propose_shared_training(
            title="Test Model",
            description="A test federated training proposal",
            resource_request={"cpu_cores": 4, "gpu_count": 1},
            training_config={"model_name": "test", "total_steps": 100},
            deadline_hours=1.0,
        )
        assert result is not None
        assert result.proposal_id

    @pytest.mark.asyncio
    async def test_vote_on_proposal_approve(self):
        fed = Federation(cluster_id="fed-node-1", cluster_endpoint="http://localhost:18001")
        prop = await fed.propose_shared_training(
            title="Test Model",
            description="A test",
            resource_request={"cpu_cores": 4},
            training_config={"model_name": "test"},
            deadline_hours=1.0,
        )
        result = await fed.vote_on_proposal(prop.proposal_id, "approve")
        assert result is True

    @pytest.mark.asyncio
    async def test_vote_on_proposal_reject(self):
        fed = Federation(cluster_id="fed-node-1", cluster_endpoint="http://localhost:18001")
        prop = await fed.propose_shared_training(
            title="Test Model",
            description="A test",
            resource_request={"cpu_cores": 4},
            training_config={"model_name": "test"},
            deadline_hours=1.0,
        )
        result = await fed.vote_on_proposal(prop.proposal_id, "reject")
        assert result is True

    @pytest.mark.asyncio
    async def test_vote_on_nonexistent_proposal(self):
        fed = Federation(cluster_id="fed-node-1", cluster_endpoint="http://localhost:18001")
        result = await fed.vote_on_proposal("nonexistent", "approve")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_status(self):
        fed = Federation(cluster_id="fed-node-1", cluster_endpoint="http://localhost:18001")
        status = fed.get_status()
        assert isinstance(status, dict)

    @pytest.mark.asyncio
    async def test_register_peer_overwrites_existing(self):
        fed = Federation(cluster_id="fed-node-1", cluster_endpoint="http://localhost:18001")
        node1 = FederationNode(node_id="peer-1", name="V1", endpoint="http://peer1:8000", resources={"cpu_cores": 4})
        node2 = FederationNode(node_id="peer-1", name="V2", endpoint="http://peer1-new:8000", resources={"cpu_cores": 16})
        await fed.register_peer(node1)
        assert fed.peers["peer-1"].name == "V1"
        await fed.register_peer(node2)
        assert fed.peers["peer-1"].name == "V2"


class TestInferenceRouterEdgeCases:
    """Test inference router edge cases."""

    def test_node_at_capacity_not_available(self):
        node = InferenceNode(
            node_id="full-node",
            endpoint="http://full:8000",
            status=InferenceStatus.READY,
            capacity=10,
            current_load=10,
            last_heartbeat=time.time(),
        )
        assert node.is_available is False

    def test_node_low_health_not_available(self):
        node = InferenceNode(
            node_id="sick-node",
            endpoint="http://sick:8000",
            status=InferenceStatus.READY,
            capacity=10,
            current_load=0,
            health_score=0.1,
            last_heartbeat=time.time(),
        )
        assert node.is_available is False

    def test_node_stale_heartbeat_not_available(self):
        node = InferenceNode(
            node_id="stale-node",
            endpoint="http://stale:8000",
            status=InferenceStatus.READY,
            capacity=10,
            current_load=0,
            last_heartbeat=0,
        )
        assert node.is_available is False

    def test_node_healthy_available(self):
        node = InferenceNode(
            node_id="healthy",
            endpoint="http://healthy:8000",
            status=InferenceStatus.READY,
            capacity=10,
            current_load=3,
            health_score=0.9,
            last_heartbeat=time.time(),
        )
        assert node.is_available is True

    def test_round_robin_cycling(self):
        lb = InferenceLoadBalancer(strategy=RoutingStrategy.ROUND_ROBIN)
        n1 = InferenceNode(node_id="n1", endpoint="http://n1:8000", status=InferenceStatus.READY, capacity=10, last_heartbeat=time.time())
        n2 = InferenceNode(node_id="n2", endpoint="http://n2:8000", status=InferenceStatus.READY, capacity=10, last_heartbeat=time.time())
        lb.register_node(n1)
        lb.register_node(n2)
        lb.add_model_to_node("n1", "test")
        lb.add_model_to_node("n2", "test")
        req = InferenceRequest(model_id="test", prompt="hi", max_tokens=1)
        first = lb.route_request(req)
        second = lb.route_request(req)
        assert first != second

    def test_least_loaded_routing(self):
        lb = InferenceLoadBalancer(strategy=RoutingStrategy.LEAST_LOADED)
        n1 = InferenceNode(node_id="n1", endpoint="http://n1:8000", status=InferenceStatus.READY, capacity=10, current_load=5, last_heartbeat=time.time())
        n2 = InferenceNode(node_id="n2", endpoint="http://n2:8000", status=InferenceStatus.READY, capacity=10, current_load=1, last_heartbeat=time.time())
        lb.register_node(n1)
        lb.register_node(n2)
        lb.add_model_to_node("n1", "test")
        lb.add_model_to_node("n2", "test")
        req = InferenceRequest(model_id="test", prompt="hi", max_tokens=1)
        result = lb.route_request(req)
        assert result == "n2"

    def test_no_nodes_returns_none(self):
        lb = InferenceLoadBalancer()
        req = InferenceRequest(model_id="test", prompt="hi", max_tokens=1)
        result = lb.route_request(req)
        assert result is None

    def test_unregister_node(self):
        lb = InferenceLoadBalancer()
        lb.register_node(InferenceNode(node_id="n1", endpoint="http://n1:8000", status=InferenceStatus.READY, capacity=10, last_heartbeat=time.time()))
        lb.unregister_node("n1")
        assert "n1" not in lb.nodes

    def test_load_factor(self):
        node = InferenceNode(node_id="n1", endpoint="http://n1:8000", status=InferenceStatus.READY, capacity=10, current_load=5, last_heartbeat=time.time())
        assert node.load_factor == 0.5

    def test_load_factor_zero_capacity(self):
        node = InferenceNode(node_id="n1", endpoint="http://n1:8000", status=InferenceStatus.READY, capacity=0, current_load=0, last_heartbeat=time.time())
        assert node.load_factor == 0.0

    def test_hash_based_routing_deterministic(self):
        lb = InferenceLoadBalancer(strategy=RoutingStrategy.HASH_BASED)
        lb.register_node(InferenceNode(node_id="n1", endpoint="http://n1:8000", status=InferenceStatus.READY, capacity=10, last_heartbeat=time.time()))
        lb.register_node(InferenceNode(node_id="n2", endpoint="http://n2:8000", status=InferenceStatus.READY, capacity=10, last_heartbeat=time.time()))
        lb.add_model_to_node("n1", "test")
        lb.add_model_to_node("n2", "test")
        req = InferenceRequest(model_id="test", prompt="hi", max_tokens=1, user_id="user123")
        first = lb.route_request(req)
        second = lb.route_request(req)
        assert first == second

    def test_group_id_filtering(self):
        lb = InferenceLoadBalancer()
        lb.register_node(InferenceNode(node_id="n1", endpoint="http://n1:8000", status=InferenceStatus.READY, capacity=10, current_load=0, group_id="g1", last_heartbeat=time.time()))
        lb.register_node(InferenceNode(node_id="n2", endpoint="http://n2:8000", status=InferenceStatus.READY, capacity=10, current_load=0, group_id="g2", last_heartbeat=time.time()))
        lb.register_node(InferenceNode(node_id="n3", endpoint="http://n3:8000", status=InferenceStatus.READY, capacity=10, current_load=0, group_id="", last_heartbeat=time.time()))
        lb.add_model_to_node("n1", "test")
        lb.add_model_to_node("n2", "test")
        lb.add_model_to_node("n3", "test")
        req_g1 = InferenceRequest(model_id="test", prompt="hi", max_tokens=1, group_id="g1")
        result = lb.route_request(req_g1)
        assert result in ("n1", "n3")


class TestGradientSyncEdgeCases:
    """Test gradient sync additional edge cases."""

    @pytest.mark.asyncio
    async def test_receive_gradients_plain_arrays(self):
        server = GradientSyncServer(node_id="test")
        result = await server.receive_gradients({
            "job_id": "j1",
            "step": 1,
            "node_id": "n1",
            "gradients": {
                "layer1": [1.0, 2.0, 3.0],
                "layer2": [4.0, 5.0, 6.0],
            },
        })
        assert result is True

    @pytest.mark.asyncio
    async def test_receive_gradients_empty(self):
        server = GradientSyncServer(node_id="test")
        result = await server.receive_gradients({
            "job_id": "j3",
            "step": 1,
            "node_id": "n1",
            "gradients": {},
        })
        assert result is False

    @pytest.mark.asyncio
    async def test_pull_nonexistent_returns_none(self):
        server = GradientSyncServer(node_id="test")
        result = await server.pull_aggregated("nonexistent", 1)
        assert result is None


class TestLocalTrainerEdgeCases:
    """Test LocalTrainer additional edge cases."""

    def test_init_creates_model_on_first_step(self):
        config = TrainingConfig(model_name="test", total_steps=10, hidden_size=16, num_heads=4, num_layers=2)
        trainer = LocalTrainer(config, checkpoint_manager=None)
        assert len(trainer._model_params) == 0
        trainer._init_model()
        assert len(trainer._model_params) > 0

    def test_load_model_state_replaces_params(self):
        config = TrainingConfig(model_name="test", total_steps=10, hidden_size=16, num_heads=4, num_layers=2)
        trainer = LocalTrainer(config)
        trainer._init_model()
        state = trainer.get_model_state()
        modified = {k: v * 2 for k, v in state.items()}
        trainer.load_model_state(modified)
        new_state = trainer.get_model_state()
        for key in state:
            assert np.allclose(new_state[key], state[key] * 2), f"{key} should be doubled"

    def test_load_model_state_preserves_optimizer(self):
        config = TrainingConfig(model_name="test", total_steps=10, hidden_size=16, num_heads=4, num_layers=2)
        trainer = LocalTrainer(config)
        trainer._init_model()
        trainer._optimizer_state = {"step": 42, "momentum": {"layer1": np.ones(16)}}
        state = trainer.get_model_state()
        modified = {k: v + 0.1 for k, v in state.items()}
        trainer.load_model_state(modified, preserve_optimizer=True)
        assert trainer._optimizer_state["step"] == 42
        assert "momentum" in trainer._optimizer_state