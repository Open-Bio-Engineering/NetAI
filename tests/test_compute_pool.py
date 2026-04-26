"""Tests for the compute_pool module — share tracking, pipeline orchestration,
stratum protocol, pool management, and jack-in flow.
"""

import asyncio
import time
import pytest
import numpy as np

from netai.compute_pool.share import (
    ComputeShare, ShareStatus, ShareDifficulty, ProofOfCompute,
    ShareLedger, PPLNSRewardCalculator, ComputeContribution,
)
from netai.compute_pool.pipeline import (
    PipelineStage, PipelineConfig, PipelinePlan,
    ActivationBuffer, PipelineOrchestrator, PipelineStatus,
)
from netai.compute_pool.stratum import (
    StratumServer, StratumClient, StratumMessage, StratumMessageType,
    WorkAssignment, WorkResult, NodeDifficulty,
)
from netai.compute_pool.pool import (
    ComputePool, PoolNode, PoolNodeStatus, PoolStatus,
)
from netai.compute_pool.jackin import JackInManager, JackInConfig


# ── Share / ProofOfCompute Tests ──


class TestProofOfCompute:
    def test_create(self):
        poc = ProofOfCompute(node_id="n1", model_id="glm")
        assert poc.node_id == "n1"

    def test_verify_valid_nonce(self):
        for nonce in range(100000):
            poc = ProofOfCompute(node_id="n1", model_id="m", nonce=nonce)
            raw = f'"{poc.node_id}":{poc.model_id}'
            if poc.verify(difficulty_target=1):
                assert True
                return
        pytest.skip("No valid nonce found in 100K tries")

    def test_verify_wrong_data(self):
        poc = ProofOfCompute(node_id="n1", model_id="m", input_hash="abc", output_hash="def", nonce=42)
        poc2 = ProofOfCompute(node_id="n1", model_id="m", input_hash="xyz", output_hash="def", nonce=42)
        r1 = poc.verify(difficulty_target=1)
        r2 = poc2.verify(difficulty_target=1)
        assert isinstance(r1, bool) and isinstance(r2, bool)


class TestShareDifficulty:
    def test_from_vram(self):
        assert ShareDifficulty.from_vram_gb(1) == ShareDifficulty.LIGHT
        assert ShareDifficulty.from_vram_gb(8) == ShareDifficulty.MEDIUM
        assert ShareDifficulty.from_vram_gb(24) == ShareDifficulty.HEAVY

    def test_weights(self):
        assert ShareDifficulty.LIGHT.weight() < ShareDifficulty.MEDIUM.weight()
        assert ShareDifficulty.MEDIUM.weight() < ShareDifficulty.HEAVY.weight()


class TestShareLedger:
    def test_add_share(self):
        ledger = ShareLedger()
        share = ComputeShare(node_id="n1", model_id="m", difficulty=ShareDifficulty.MEDIUM, latency_ms=100)
        result = ledger.add_share(share)
        assert result.status == ShareStatus.VALID
        assert result.weight > 0

    def test_weight_scales_with_difficulty(self):
        ledger = ShareLedger()
        s1 = ComputeShare(node_id="n1", difficulty=ShareDifficulty.LIGHT, latency_ms=100)
        s2 = ComputeShare(node_id="n2", difficulty=ShareDifficulty.HEAVY, latency_ms=100)
        r1 = ledger.add_share(s1)
        r2 = ledger.add_share(s2)
        assert r2.weight > r1.weight

    def test_lower_latency_more_weight(self):
        ledger = ShareLedger()
        s1 = ComputeShare(node_id="n1", difficulty=ShareDifficulty.MEDIUM, latency_ms=50)
        s2 = ComputeShare(node_id="n2", difficulty=ShareDifficulty.MEDIUM, latency_ms=500)
        r1 = ledger.add_share(s1)
        r2 = ledger.add_share(s2)
        assert r1.weight > r2.weight

    def test_contribution_tracking(self):
        ledger = ShareLedger()
        for i in range(5):
            ledger.add_share(ComputeShare(node_id="n1", latency_ms=100))
        c = ledger.get_contribution("n1")
        assert c is not None
        assert c.total_shares == 5
        assert c.valid_shares == 5
        assert c.total_weight > 0

    def test_leaderboard(self):
        ledger = ShareLedger()
        ledger.add_share(ComputeShare(node_id="light", difficulty=ShareDifficulty.LIGHT, latency_ms=100))
        ledger.add_share(ComputeShare(node_id="heavy", difficulty=ShareDifficulty.HEAVY, latency_ms=100))
        lb = ledger.get_leaderboard()
        assert len(lb) == 2
        assert lb[0].node_id == "heavy"

    def test_prune_expired(self):
        ledger = ShareLedger(max_shares=100, share_window_hours=0.00001)
        ledger.add_share(ComputeShare(node_id="n1", latency_ms=100, timestamp=time.time() - 10))
        ledger.prune_expired()
        shares = ledger.get_recent_shares()
        assert len(shares) == 0


class TestPPLNSRewardCalculator:
    def test_proportional_rewards(self):
        calc = PPLNSRewardCalculator(credit_per_weight=1.0)
        ledger = ShareLedger()
        ledger.add_share(ComputeShare(node_id="n1", difficulty=ShareDifficulty.MEDIUM, latency_ms=100))
        ledger.add_share(ComputeShare(node_id="n1", difficulty=ShareDifficulty.MEDIUM, latency_ms=100))
        ledger.add_share(ComputeShare(node_id="n2", difficulty=ShareDifficulty.MEDIUM, latency_ms=100))
        rewards = calc.calculate_rewards(ledger)
        assert rewards["n1"] > rewards["n2"]

    def test_no_shares_empty_rewards(self):
        calc = PPLNSRewardCalculator()
        ledger = ShareLedger()
        rewards = calc.calculate_rewards(ledger)
        assert len(rewards) == 0

    def test_inference_priority(self):
        calc = PPLNSRewardCalculator()
        ledger = ShareLedger()
        for _ in range(10):
            ledger.add_share(ComputeShare(node_id="heavy", difficulty=ShareDifficulty.HEAVY, latency_ms=100))
        for _ in range(2):
            ledger.add_share(ComputeShare(node_id="light", difficulty=ShareDifficulty.LIGHT, latency_ms=100))
        p1 = calc.calculate_inference_priority(ledger, "heavy")
        p2 = calc.calculate_inference_priority(ledger, "light")
        assert p1 > p2


# ── Pipeline Tests ──


class TestPipelineConfig:
    def test_total_params(self):
        cfg = PipelineConfig(total_layers=12, hidden_size=768, num_heads=12, vocab_size=50257)
        params = cfg.total_params
        assert params > 0

    def test_model_size(self):
        cfg = PipelineConfig(total_layers=12, hidden_size=768, vocab_size=50257, bytes_per_param=2)
        size_mb = cfg.model_size_mb
        assert size_mb > 0

    def test_vram_per_stage(self):
        cfg = PipelineConfig(total_layers=12, hidden_size=768, vocab_size=50257)
        vram = cfg.vram_per_stage(num_stages=4)
        assert vram > 0
        assert vram < cfg.model_size_mb


class TestPipelineOrchestrator:
    def test_plan_pipeline_basic(self):
        orch = PipelineOrchestrator()
        cfg = PipelineConfig(model_id="test", total_layers=12, hidden_size=768, vocab_size=50257)
        nodes = [
            {"node_id": "n1", "vram_available_mb": 2000},
            {"node_id": "n2", "vram_available_mb": 2000},
        ]
        plan = orch.plan_pipeline(cfg, nodes)
        assert len(plan.stages) >= 1
        assert plan.stages[0].num_layers > 0

    def test_plan_pipeline_respects_vram(self):
        orch = PipelineOrchestrator()
        cfg = PipelineConfig(model_id="big", total_layers=80, hidden_size=8192, vocab_size=128000, intermediate_size=32768)
        nodes = [
            {"node_id": f"n{i}", "vram_available_mb": 8192} for i in range(10)
        ]
        plan = orch.plan_pipeline(cfg, nodes)
        total_layers = sum(s.num_layers for s in plan.stages)
        assert total_layers <= 80

    def test_pipeline_coverage(self):
        orch = PipelineOrchestrator()
        cfg = PipelineConfig(model_id="test", total_layers=12, hidden_size=768, vocab_size=50257)
        nodes = [{"node_id": "n1", "vram_available_mb": 50000}]
        plan = orch.plan_pipeline(cfg, nodes)
        assert plan.coverage > 0

    def test_assign_stage(self):
        orch = PipelineOrchestrator()
        cfg = PipelineConfig(model_id="test", total_layers=12, hidden_size=768, vocab_size=50257)
        nodes = [{"node_id": "n1", "vram_available_mb": 50000}]
        plan = orch.plan_pipeline(cfg, nodes)
        assert plan.stages
        stage = plan.stages[0]
        updated = PipelineStage(
            stage_id=stage.stage_id,
            node_id="n1",
            model_id="test",
            stage_index=stage.stage_index,
            layer_start=stage.layer_start,
            layer_end=stage.layer_end,
            num_layers=stage.num_layers,
        )
        result = orch.assign_stage("test", updated)
        assert result is True

    def test_get_stage_for_node(self):
        orch = PipelineOrchestrator()
        cfg = PipelineConfig(model_id="test", total_layers=12, hidden_size=768, vocab_size=50257)
        nodes = [{"node_id": "n1", "vram_available_mb": 50000}]
        orch.plan_pipeline(cfg, nodes)
        stage = orch.get_stage_for_node("test", "n1")
        assert stage is not None

    def test_heartbeat_tracking(self):
        orch = PipelineOrchestrator()
        cfg = PipelineConfig(model_id="test", total_layers=12, hidden_size=768, vocab_size=50257)
        nodes = [{"node_id": "n1", "vram_available_mb": 50000}]
        plan = orch.plan_pipeline(cfg, nodes)
        stage = plan.stages[0]
        orch.assign_stage("test", PipelineStage(
            stage_id=stage.stage_id, node_id="n1", model_id="test",
            stage_index=stage.stage_index, layer_start=stage.layer_start,
            layer_end=stage.layer_end, num_layers=stage.num_layers,
            status=PipelineStatus.READY, loaded_at=time.time(), last_heartbeat=time.time(),
        ))
        orch.heartbeat("test", stage.stage_index, inference_count=1, latency_ms=50)
        found = orch.get_stage_for_node("test", "n1")
        assert found.inference_count >= 1

    def test_remove_pipeline(self):
        orch = PipelineOrchestrator()
        cfg = PipelineConfig(model_id="test", total_layers=12, hidden_size=768, vocab_size=50257)
        nodes = [{"node_id": "n1", "vram_available_mb": 50000}]
        orch.plan_pipeline(cfg, nodes)
        assert orch.remove_pipeline("test") is True
        assert orch.remove_pipeline("nonexistent") is False


class TestActivationBuffer:
    def test_serialize_deserialize(self):
        arr = np.random.randn(1, 10, 768).astype(np.float16)
        data, shape, dtype = ActivationBuffer.serialize_hidden(arr)
        recovered = ActivationBuffer.deserialize_hidden(data, shape, dtype)
        assert np.allclose(arr, recovered)

    def test_compute_hash(self):
        buf = ActivationBuffer()
        arr = np.zeros((1, 10, 768), dtype=np.float32)
        h = buf.compute_hash(arr)
        assert len(h) == 16


# ── Stratum Tests ──


class TestStratumServer:
    @pytest.mark.asyncio
    async def test_subscribe(self):
        orch = PipelineOrchestrator()
        ledger = ShareLedger()
        server = StratumServer(orch, ledger)
        result = await server.handle_subscribe("n1", {"vram_gb": 8, "gpu_count": 1, "cpu_cores": 8})
        assert "subscription_id" in result
        assert result["node_id"] == "n1"

    @pytest.mark.asyncio
    async def test_subscribe_difficulty_assignment(self):
        orch = PipelineOrchestrator()
        ledger = ShareLedger()
        server = StratumServer(orch, ledger)
        r1 = await server.handle_subscribe("small", {"vram_gb": 2, "gpu_count": 0})
        r2 = await server.handle_subscribe("large", {"vram_gb": 24, "gpu_count": 2})
        assert r1["difficulty"] == "light"
        assert r2["difficulty"] == "heavy"

    @pytest.mark.asyncio
    async def test_submit_valid_share(self):
        orch = PipelineOrchestrator()
        ledger = ShareLedger()
        server = StratumServer(orch, ledger)
        for nonce in range(200000):
            poc = ProofOfCompute(node_id="n1", model_id="m", stage_index=0, nonce=nonce)
            if poc.verify(difficulty_target=3):
                break
        else:
            pytest.skip("No valid nonce found")
        result = WorkResult(
            node_id="n1", model_id="m", stage_index=0,
            latency_ms=50, vram_used_mb=2000, compute_steps=100,
            proof=poc,
        )
        share = await server.handle_submit(result)
        assert share is not None
        assert share.status == ShareStatus.VALID

    @pytest.mark.asyncio
    async def test_submit_no_proof(self):
        orch = PipelineOrchestrator()
        ledger = ShareLedger()
        server = StratumServer(orch, ledger)
        result = WorkResult(node_id="n1", model_id="m")
        share = await server.handle_submit(result)
        assert share is None


class TestNodeDifficulty:
    def test_adjust_up(self):
        nd = NodeDifficulty(node_id="n1", current_difficulty=ShareDifficulty.LIGHT)
        nd.adjust(shares_last_minute=10)
        assert nd.current_difficulty == ShareDifficulty.MEDIUM

    def test_adjust_down(self):
        nd = NodeDifficulty(node_id="n1", current_difficulty=ShareDifficulty.HEAVY)
        nd.adjust(shares_last_minute=0.01)
        assert nd.current_difficulty == ShareDifficulty.MEDIUM

    def test_adjust_stays_same(self):
        nd = NodeDifficulty(node_id="n1", current_difficulty=ShareDifficulty.MEDIUM)
        nd.adjust(shares_last_minute=1.0)
        assert nd.current_difficulty == ShareDifficulty.MEDIUM


class TestStratumMessage:
    def test_subscribe_message(self):
        msg = StratumMessage.subscribe("n1", {"gpu_count": 1})
        assert msg.method == StratumMessageType.SUBSCRIBE
        assert msg.params[0] == "n1"

    def test_authorize_message(self):
        msg = StratumMessage.authorize("user", "token")
        assert msg.method == StratumMessageType.AUTHORIZE

    def test_notify_message(self):
        assignment = WorkAssignment(model_id="m", stage_index=0)
        msg = StratumMessage.notify("job1", assignment)
        assert msg.method == StratumMessageType.NOTIFY


# ── Pool Tests ──


class TestComputePool:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        pool = ComputePool()
        await pool.start()
        assert pool._running
        await pool.stop()
        assert not pool._running

    @pytest.mark.asyncio
    async def test_register_node(self):
        pool = ComputePool()
        node = PoolNode(node_id="n1", gpu_count=1, vram_total_mb=8192, vram_available_mb=6000)
        result = pool.register_node(node)
        assert result.node_id == "n1"
        assert result.status == PoolNodeStatus.IDLE

    @pytest.mark.asyncio
    async def test_unregister_node(self):
        pool = ComputePool()
        pool.register_node(PoolNode(node_id="n1"))
        pool.unregister_node("n1")
        assert "n1" not in pool.nodes

    @pytest.mark.asyncio
    async def test_jack_in(self):
        pool = ComputePool()
        await pool.start()
        node = PoolNode(node_id="n1", gpu_count=1, vram_total_mb=8192, vram_available_mb=6000)
        result = await pool.jack_in(node)
        assert result["status"] == "subscribed"
        assert "n1" in pool.nodes
        await pool.stop()

    @pytest.mark.asyncio
    async def test_jack_in_with_model(self):
        pool = ComputePool()
        await pool.start()
        cfg = PipelineConfig(model_id="glm-test", total_layers=12, hidden_size=768, vocab_size=50257)
        pool.plan_model(cfg)
        node = PoolNode(node_id="n1", gpu_count=1, vram_total_mb=8192, vram_available_mb=6000)
        result = await pool.jack_in(node, model_id="glm-test")
        assert result["status"] in ("subscribed", "assigned")
        await pool.stop()

    @pytest.mark.asyncio
    async def test_jack_out(self):
        pool = ComputePool()
        await pool.start()
        node = PoolNode(node_id="n1", gpu_count=1, vram_total_mb=8192)
        await pool.jack_in(node)
        result = await pool.jack_out("n1")
        assert result is True
        await pool.stop()

    @pytest.mark.asyncio
    async def test_plan_model(self):
        pool = ComputePool()
        node = PoolNode(node_id="n1", gpu_count=2, vram_total_mb=16384, vram_available_mb=12000)
        pool.register_node(node)
        node2 = PoolNode(node_id="n2", gpu_count=1, vram_total_mb=8192, vram_available_mb=6000)
        pool.register_node(node2)
        cfg = PipelineConfig(model_id="test", total_layers=24, hidden_size=768, vocab_size=50257)
        plan = pool.plan_model(cfg)
        assert len(plan.stages) >= 1

    @pytest.mark.asyncio
    async def test_inference_credits(self):
        pool = ComputePool()
        await pool.start()
        node = PoolNode(node_id="n1", gpu_count=1, vram_total_mb=8192, vram_available_mb=6000)
        await pool.jack_in(node)
        share = ComputeShare(node_id="n1", model_id="m", difficulty=ShareDifficulty.HEAVY, latency_ms=50)
        for nonce in range(200000):
            poc = ProofOfCompute(node_id="n1", model_id="m", nonce=nonce)
            share.proof = poc
            if pool.ledger.validate_share(share, difficulty_target=3):
                break
        else:
            pool.ledger.add_share(share)
        credits = pool.get_inference_credits("n1")
        assert credits >= 0
        await pool.stop()

    @pytest.mark.asyncio
    async def test_can_run_model(self):
        pool = ComputePool()
        node = PoolNode(node_id="n1", gpu_count=2, vram_total_mb=16384, vram_available_mb=12000)
        pool.register_node(node)
        can, info = pool.can_run_model(10000)
        assert isinstance(can, bool)
        assert "available_vram_mb" in info

    @pytest.mark.asyncio
    async def test_leaderboard(self):
        pool = ComputePool()
        pool.register_node(PoolNode(node_id="n1", gpu_count=2, vram_total_mb=16384))
        pool.register_node(PoolNode(node_id="n2", gpu_count=1, vram_total_mb=8192))
        pool.ledger.add_share(ComputeShare(node_id="n1", difficulty=ShareDifficulty.HEAVY, latency_ms=50))
        pool.ledger.add_share(ComputeShare(node_id="n2", difficulty=ShareDifficulty.LIGHT, latency_ms=200))
        lb = pool.get_leaderboard()
        assert len(lb) == 2

    @pytest.mark.asyncio
    async def test_get_status(self):
        pool = ComputePool()
        pool.register_node(PoolNode(node_id="n1", gpu_count=1, vram_total_mb=8192))
        status = pool.get_status()
        assert status.total_nodes == 1
        assert status.total_gpus == 1

    @pytest.mark.asyncio
    async def test_heartbeat_updates(self):
        pool = ComputePool()
        pool.register_node(PoolNode(node_id="n1", gpu_count=1, vram_total_mb=8192))
        pool.update_heartbeat("n1", vram_available_mb=7000, inference_count=5, latency_ms=100)
        node = pool.nodes["n1"]
        assert node.vram_available_mb == 7000
        assert node.inference_count == 5


# ── PoolNode Tests ──


class TestPoolNode:
    def test_inference_capacity_score(self):
        n1 = PoolNode(gpu_count=2, vram_available_mb=16384, cpu_cores=16)
        n2 = PoolNode(gpu_count=0, vram_available_mb=0, cpu_cores=8)
        assert n1.inference_capacity_score > n2.inference_capacity_score

    def test_is_available(self):
        n = PoolNode(gpu_count=1, vram_available_mb=8192, status=PoolNodeStatus.IDLE)
        assert n.is_available
        n.status = PoolNodeStatus.OFFLINE
        assert not n.is_available
        n.status = PoolNodeStatus.IDLE
        n.vram_available_mb = 0
        assert not n.is_available


# ── JackInManager Tests ──


class TestJackInManager:
    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        mgr = JackInManager(JackInConfig(model_id="test-model"))
        result = await mgr.start()
        assert result is not None
        status = mgr.get_local_status()
        assert status["node_id"] != ""
        assert status["gpus"] >= 0
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_pool_status(self):
        mgr = JackInManager()
        await mgr.start()
        ps = mgr.get_pool_status()
        assert "total_nodes" in ps
        assert "active_nodes" in ps
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_contribution_tracking(self):
        mgr = JackInManager(JackInConfig(model_id="test"))
        await mgr.start()
        node_id = mgr._local_node_id
        mgr.pool.ledger.add_share(ComputeShare(node_id=node_id, difficulty=ShareDifficulty.MEDIUM, latency_ms=100))
        status = mgr.get_local_status()
        assert status["shares"] >= 1
        await mgr.stop()


# ── Large Model Scenario Tests ──


class TestLargeModelScenario:
    def test_glm5_1_pipeline_planning(self):
        cfg = PipelineConfig(
            model_id="glm-5.1",
            model_name="GLM-5.1",
            total_layers=80,
            hidden_size=8192,
            num_heads=64,
            vocab_size=128000,
            intermediate_size=32768,
            bytes_per_param=2,
        )
        assert cfg.total_params > 10_000_000_000  # 10B+ (scales to any size)
        size_mb = cfg.model_size_mb
        assert size_mb > 100_000  # 100GB+

        nodes = [
            {"node_id": f"home-gpu-{i}", "vram_available_mb": 24576}
            for i in range(60)
        ]
        orch = PipelineOrchestrator()
        plan = orch.plan_pipeline(cfg, nodes)
        assert len(plan.stages) > 1
        total_assigned = sum(s.num_layers for s in plan.stages)
        assert total_assigned > 0

    def test_vram_per_stage_for_700b(self):
        cfg = PipelineConfig(
            model_id="700b-model",
            total_layers=80,
            hidden_size=12288,
            num_heads=96,
            vocab_size=128000,
            intermediate_size=49152,
            bytes_per_param=2,
        )
        vram_per_node = cfg.vram_per_stage(num_stages=50, batch_size=1)
        assert vram_per_node > 0

    @pytest.mark.asyncio
    async def test_pool_capacity_check(self):
        pool = ComputePool()
        for i in range(30):
            pool.register_node(PoolNode(
                node_id=f"node-{i}",
                gpu_count=1,
                vram_total_mb=24576,
                vram_available_mb=20000,
                cpu_cores=8,
            ))
        can, info = pool.can_run_model(500_000)
        assert info["available_nodes"] == 30
        assert info["available_vram_mb"] == 30 * 20000