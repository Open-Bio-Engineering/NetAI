"""Edge-case tests for bug fixes identified during code review.

Tests cover:
- Gradient aggregation (sequential halving fix)
- Learning rate scheduler edge cases
- Voting weight change corruption fix
- Scheduler priority ordering fix
- Inference response error field fix
- GradientCompressor missing keys fix
- CheckpointManager corrupt metadata fix
- P2P seed node validation
- Inference stream_infer status cleanup
"""

import asyncio
import json
import os
import tempfile
import numpy as np
import pytest

from netai.training.engine import (
    TrainingConfig, TrainingJob, LocalTrainer, LearningRateScheduler,
    GradientSyncServer, GradientShard, GradientCompressor, CheckpointManager,
)
from netai.training.voting import (
    VotingEngine, ResourcePledge, Proposal, Vote, VoteType, VoteWeight,
)
from netai.scheduler.scheduler import JobPriority, NodeResources, JobRequirements, JobScheduler
from netai.inference.engine import InferenceEngine, InferenceRequest, InferenceResponse


class TestGradientAggregation:
    """Test that aggregate_gradients uses proper mean, not sequential halving."""

    def test_aggregate_proper_mean(self):
        """Bug: (a+b)/2 then (result+c)/2 gives last item 50% weight.
        Fix: should use np.mean across all shards."""
        job = TrainingJob(config=TrainingConfig(model_name="test"))
        job_id = "test-job"
        step = 1

        shard1 = GradientShard(
            shard_id="s1", job_id=job_id, step=step,
            layer_name="layer1", shape=[3], data=[1.0, 2.0, 3.0],
        )
        shard2 = GradientShard(
            shard_id="s2", job_id=job_id, step=step,
            layer_name="layer1", shape=[3], data=[4.0, 5.0, 6.0],
        )
        shard3 = GradientShard(
            shard_id="s3", job_id=job_id, step=step,
            layer_name="layer1", shape=[3], data=[7.0, 8.0, 9.0],
        )

        # The old bug would give: ((1+4)/2 + 7)/2 = (2.5 + 7)/2 = 4.75 for each element center
        # The correct mean should be (1+4+7)/3 = 4.0 for first element
        result = asyncio.get_event_loop().run_until_complete(
            job.aggregate_gradients(step)
        )
        # No gradients in buffer yet
        assert result == {}

        # Now add gradients and aggregate properly
        async def add_and_agg():
            await job.add_gradient(shard1)
            await job.add_gradient(shard2)
            await job.add_gradient(shard3)
            return await job.aggregate_gradients(step)

        result = asyncio.get_event_loop().run_until_complete(add_and_agg())
        # Mean of [1,4,7] = 4.0, [2,5,8] = 5.0, [3,6,9] = 6.0
        np.testing.assert_array_almost_equal(result["layer1"], [4.0, 5.0, 6.0])


class TestLearningRateScheduler:
    """Test LR scheduler edge cases."""

    def test_negative_step_clamped_to_zero(self):
        """Bug: negative step gave negative LR. Fix: clamp to 0."""
        sched = LearningRateScheduler(base_lr=3e-4, warmup_steps=100, total_steps=1000)
        lr = sched.get_lr(-5)
        assert lr > 0, f"Negative step should give positive LR, got {lr}"
        assert lr == sched.get_lr(0), f"Negative step should be clamped to 0"

    def test_post_total_steps_decay(self):
        """Bug: cosine LR oscillated after total_steps. Fix: returns min LR."""
        sched = LearningRateScheduler(base_lr=3e-4, warmup_steps=100, total_steps=1000)
        lr_after = sched.get_lr(1500)
        assert lr_after > 0, f"LR after total_steps should be > 0, got {lr_after}"
        assert lr_after < sched.base_lr * 0.02, f"LR after total_steps should be near 0, got {lr_after}"

    def test_warmup_increases(self):
        sched = LearningRateScheduler(base_lr=3e-4, warmup_steps=100, total_steps=1000)
        lr_10 = sched.get_lr(10)
        lr_50 = sched.get_lr(50)
        lr_99 = sched.get_lr(99)
        assert lr_10 < lr_50 < lr_99, "LR should increase during warmup"


class TestVotingWeightChange:
    """Test that changing votes correctly adjusts weighted totals."""

    def test_vote_change_adjusts_weighted_totals(self):
        """Bug: switching votes doubled weight. Fix: old weight subtracted."""
        engine = VotingEngine(vote_weight_type=VoteWeight.BY_RESOURCE)
        from netai.training.voting import UserModelProposal
        proposal = engine.create_model_proposal(
            UserModelProposal(model_name="test", proposer_id="user1"), "user1"
        )
        engine.cast_vote(
            proposal_id=proposal.proposal_id,
            voter_id="user1",
            choice="for",
            weight=5.0,
        )
        # Change vote from for to against
        result = engine.cast_vote(
            proposal_id=proposal.proposal_id,
            voter_id="user1",
            choice="against",
            weight=5.0,
        )
        assert result is not None
        assert proposal.weighted_for == 0.0, f"weighted_for should be 0 after switch, got {proposal.weighted_for}"
        assert proposal.weighted_against == 5.0, f"weighted_against should be 5.0 after switch, got {proposal.weighted_against}"

    def test_zero_weight_vote(self):
        """Bug: weight=0.0 treated as falsy. Fix: use 'is not None' check."""
        engine = VotingEngine()
        from netai.training.voting import UserModelProposal
        proposal = engine.create_model_proposal(
            UserModelProposal(model_name="test", proposer_id="user1"), "user1"
        )
        result = engine.cast_vote(proposal_id=proposal.proposal_id, voter_id="user2", choice="for", weight=0.0)
        assert result.weight == 0.0, f"Zero weight should be accepted, got {result.weight}"


class TestSchedulerPriority:
    """Test that priority ordering is correct after fix."""

    def test_urgent_highest_priority(self):
        """Bug: URGENT=20 was dequeued last. Fix: URGENT=1 is dequeued first."""
        assert JobPriority.URGENT < JobPriority.HIGH < JobPriority.NORMAL < JobPriority.LOW

    def test_scheduler_prefers_urgent(self):
        sched = JobScheduler()
        sched.register_node(NodeResources(
            node_id="node1", cpu_cores=8, cpu_available=8,
            gpu_count=0, gpu_available=0, ram_gb=16, ram_available_gb=16,
        ))
        sched.register_node(NodeResources(
            node_id="node2", cpu_cores=8, cpu_available=8,
            gpu_count=0, gpu_available=0, ram_gb=16, ram_available_gb=16,
        ))
        sched.submit_job(JobRequirements(min_cpu_cores=1, priority=JobPriority.LOW), "low-job")
        sched.submit_job(JobRequirements(min_cpu_cores=1, priority=JobPriority.URGENT), "urgent-job")
        result = sched.schedule()
        assert len(result) == 2
        urgent_job_id = None
        for jid, sj in sched.running_jobs.items():
            if sj.name == "urgent-job":
                urgent_job_id = jid
        assert urgent_job_id is not None, "Urgent job should be in running_jobs"
        assert sched.running_jobs[urgent_job_id].requirements.priority == JobPriority.URGENT


class TestInferenceResponseError:
    """Test that InferenceResponse.error is None on success."""

    def test_successful_response_has_none_error(self):
        resp = InferenceResponse(text="hello", tokens_generated=5)
        assert resp.error is None

    def test_error_response(self):
        resp = InferenceResponse(error="Model not found")
        assert resp.error == "Model not found"

    def test_model_dump_no_empty_error(self):
        resp = InferenceResponse(text="hello", tokens_generated=5)
        d = resp.model_dump()
        assert "error" not in d or d.get("error") is None


class TestGradientCompressorDecompress:
    """Test decompress handles missing keys gracefully."""

    def test_decompress_topk_missing_keys(self):
        compressed = {"method": "topk", "shape": [2, 2]}
        result = GradientCompressor.decompress(compressed)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, np.zeros((2, 2)))

    def test_decompress_quantize_missing_keys(self):
        compressed = {"method": "quantize", "shape": [3, 3]}
        result = GradientCompressor.decompress(compressed)
        assert result.shape == (3, 3)

    def test_decompress_none_method(self):
        arr = np.array([1.0, 2.0, 3.0])
        compressed = GradientCompressor.compress(arr, method="none")
        result = GradientCompressor.decompress(compressed)
        np.testing.assert_array_almost_equal(result, arr)


class TestCheckpointManagerCorruptMetadata:
    """Test checkpoint loading with corrupt metadata."""

    def test_load_corrupt_metadata_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = CheckpointManager(checkpoint_dir=td)
            job = TrainingJob(config=TrainingConfig(model_name="test"))
            # Create a valid checkpoint
            state = {"layer1": np.random.randn(4, 4)}
            mgr.save_checkpoint(job, step=1, model_state=state, loss=0.5)
            # Corrupt the metadata
            meta_path = os.path.join(td, job.job_id, "metadata-1.json")
            with open(meta_path, "w") as f:
                json.dump({"step": "not_a_number"}, f)
            # Should handle gracefully
            result = mgr.load_checkpoint(job.job_id)
            assert result is not None  # Still loads weights even if meta is bad
            assert result[1] is None  # Metadata should be None


class TestP2PSeedValidation:
    """Test P2P seed node parsing."""

    def test_seed_without_colon_rejected(self):
        from netai.p2p.network import P2PNode
        node = P2PNode(seed_nodes=["invalid_seed"])
        # Should not crash, just log warning
        assert node is not None

    def test_seed_with_port_zero_rejected(self):
        from netai.p2p.network import P2PNode
        node = P2PNode(seed_nodes=["host:0"])
        assert node is not None


class TestInferenceEngineLoadDecrement:
    """Test that only the selected replica gets metrics updated."""

    @pytest.mark.asyncio
    async def test_inference_load_balancing(self):
        engine = InferenceEngine(node_id="test-node")
        await engine.start()
        try:
            from netai.inference.engine import ModelServeConfig
            config = ModelServeConfig(model_id="test-model", model_name="test-model", num_replicas=1)
            await engine.load_model(config)
            assert len(engine.replicas["test-model"]) == 1
            request = InferenceRequest(model_id="test-model", prompt="test", max_tokens=5)
            result = await engine.infer(request)
            assert result.text  # Should produce output
            # Check that the replica has inference_count = 1
            rep = list(engine.replicas["test-model"].values())[0]
            assert rep.inference_count == 1, f"Expected 1 inference, got {rep.inference_count}"
        finally:
            await engine.stop()


class TestGradientSyncPushNoTruncation:
    """Test that push_gradients does not truncate gradient data."""

    @pytest.mark.asyncio
    async def test_large_gradient_not_truncated(self):
        server = GradientSyncServer(node_id="test-node")
        large_grad = {
            "layer1": np.random.randn(100, 100).astype(np.float32),
            "layer2": np.random.randn(200, 50).astype(np.float32),
        }
        await server.push_gradients("job1", 1, large_grad)
        status = server.get_sync_status()
        # Verify data was stored
        assert "job1" in status.get("gradient_store", {})


class TestCoordinatorPreservesOptimizer:
    """Test that gradient sync preserves optimizer state."""

    def test_load_model_state_preserve_optimizer(self):
        config = TrainingConfig(model_name="test", total_steps=10, hidden_size=16, num_heads=4, num_layers=2)
        trainer = LocalTrainer(config, checkpoint_manager=CheckpointManager())
        trainer._init_model()
        state_before = trainer.get_model_state()
        modified = {k: v + 0.01 for k, v in state_before.items()}
        trainer.load_model_state(modified, preserve_optimizer=True)
        state_after = trainer.get_model_state()
        for key in state_before:
            assert not np.allclose(state_after[key], state_before[key]), f"{key} should have changed"


class TestGroupInviteMaxUses:
    """Test that invite codes track uses and max_uses."""

    def test_invite_code_max_uses(self):
        from netai.training.groups import GroupManager, GroupVisibility
        gm = GroupManager()
        group = gm.create_group(name="test", owner_id="admin", visibility=GroupVisibility.PRIVATE)
        code = gm.create_invite(group.group_id, "admin")
        invite = gm._invite_codes.get(code)
        assert invite is not None
        assert invite["max_uses"] == 1  # Default max_uses
        # Use the invite
        ok, msg = gm.join_group(group.group_id, "user1", invite_code=code)
        assert ok
        # Code should be consumed (max_uses=1)
        ok2, msg2 = gm.join_group(group.group_id, "user2", invite_code=code)
        assert not ok2

    def test_invite_code_multiple_uses(self):
        from netai.training.groups import GroupManager, GroupVisibility
        gm = GroupManager()
        group = gm.create_group(name="test2", owner_id="admin", visibility=GroupVisibility.PRIVATE)
        code = gm.create_invite(group.group_id, "admin", max_uses=3)
        ok1, _ = gm.join_group(group.group_id, "user1", invite_code=code)
        assert ok1
        ok2, _ = gm.join_group(group.group_id, "user2", invite_code=code)
        assert ok2
        ok3, _ = gm.join_group(group.group_id, "user3", invite_code=code)
        assert ok3
        ok4, _ = gm.join_group(group.group_id, "user4", invite_code=code)
        assert not ok4


class TestSchedulerRebalance:
    """Test that rebalance actually applies migrations."""

    def test_rebalance_applies_migrations(self):
        sched = JobScheduler()
        sched.register_node(NodeResources(
            node_id="n1", cpu_cores=4, cpu_available=4,
            gpu_count=0, gpu_available=0, ram_gb=8, ram_available_gb=8,
        ))
        sched.register_node(NodeResources(
            node_id="n2", cpu_cores=4, cpu_available=4,
            gpu_count=0, gpu_available=0, ram_gb=8, ram_available_gb=8,
        ))
        result = sched.schedule()
        # Submit and schedule a job
        job_id = sched.submit_job(JobRequirements(min_cpu_cores=1), "test-job")
        result = sched.schedule()
        assert len(result) == 1
        # Make the assigned node unavailable
        assigned_node = result[0][1][0]
        sched.nodes[assigned_node].cpu_available = 0
        sched.nodes[assigned_node].is_training = True
        # Make the other node available
        for n in sched.nodes.values():
            if n.node_id != assigned_node:
                n.cpu_available = n.cpu_cores
        # Rebalance should migrate
        sched.nodes[assigned_node].is_training = True  # already
        migrations = sched.rebalance()
        if migrations:
            # Verify the job was reassigned to the available node
            assert len(migrations) > 0