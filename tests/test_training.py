"""Tests for training engine, voting, groups, scheduler, GitHub integration."""

import asyncio
import json
import os
import tempfile
import time

import numpy as np
import pytest

from netai.training.engine import (
    TrainingConfig, TrainingJob, TrainingStatus, TrainingMetrics,
    GradientShard, ModelCheckpoint, LocalTrainer, CheckpointManager,
    ShardScheduler, GradientCompressor, LearningRateScheduler,
    DeviceType, OptimizerType, ShardAssignment,
)
from netai.training.voting import (
    VotingEngine, ResourcePledge, Proposal, Vote, UserModelProposal,
    VoteType, VoteWeight, ProposalStatus,
)
from netai.training.groups import (
    GroupManager, TrainingGroup, GroupVisibility, MemberRole, GroupPolicy,
)
from netai.github.integration import GitHubIntegration, GitHubConfig, WebhookEvent, CommitInfo
from netai.scheduler.scheduler import (
    JobScheduler, NodeResources, JobRequirements, JobPriority, SchedulePolicy,
)
from netai.crypto.identity import NodeIdentity, GroupKey, derive_group_key, hash_password, verify_password


# ── Training Engine ──

class TestTrainingConfig:
    def test_defaults(self):
        c = TrainingConfig()
        assert c.model_name == "gpt2-small"
        assert c.optimizer == OptimizerType.ADAMW
        assert c.total_steps == 10000
        assert c.batch_size == 8

    def test_custom(self):
        c = TrainingConfig(model_name="llama-7b", total_steps=5000, learning_rate=1e-4)
        assert c.model_name == "llama-7b"
        assert c.total_steps == 5000


class TestGradientShard:
    def test_compute_hash(self):
        s = GradientShard(shard_id="s1", job_id="j1", step=1, layer_name="l1", data=[1.0, 2.0], shape=[2])
        h = s.compute_hash()
        assert len(h) == 16

    def test_different_data_same_id(self):
        s1 = GradientShard(shard_id="s1", job_id="j1", step=1, layer_name="l1", data=[1.0], shape=[1])
        s2 = GradientShard(shard_id="s1", job_id="j1", step=1, layer_name="l1", data=[2.0], shape=[1])
        assert s1.compute_hash() != s2.compute_hash()

    def test_same_data_same_hash(self):
        s1 = GradientShard(shard_id="s1", job_id="j1", step=1, layer_name="l1", data=[1.0], shape=[1])
        s2 = GradientShard(shard_id="s1", job_id="j1", step=1, layer_name="l1", data=[1.0], shape=[1])
        assert s1.compute_hash() == s2.compute_hash()


class TestTrainingJob:
    def test_record_metrics(self):
        job = TrainingJob(TrainingConfig())
        m = TrainingMetrics(step=1, loss=5.0, lr=3e-4)
        job.record_metrics(m)
        assert job.current_step == 1
        assert job.best_loss == 5.0

    def test_best_loss_tracking(self):
        job = TrainingJob(TrainingConfig())
        job.record_metrics(TrainingMetrics(step=1, loss=5.0))
        job.record_metrics(TrainingMetrics(step=2, loss=4.0))
        job.record_metrics(TrainingMetrics(step=3, loss=4.5))
        assert job.best_loss == 4.0

    @pytest.mark.asyncio
    async def test_add_gradient(self):
        job = TrainingJob(TrainingConfig())
        shard = GradientShard(shard_id="s1", job_id=job.job_id, step=1, layer_name="l1",
                             data=[1.0, 2.0, 3.0], shape=[3])
        await job.add_gradient(shard)
        grads = await job.get_gradients_for_step(1)
        assert len(grads) == 1

    @pytest.mark.asyncio
    async def test_aggregate_gradients(self):
        job = TrainingJob(TrainingConfig())
        s1 = GradientShard(shard_id="s1", job_id=job.job_id, step=1, layer_name="fc1",
                           data=[2.0, 4.0], shape=[2])
        s2 = GradientShard(shard_id="s2", job_id=job.job_id, step=1, layer_name="fc1",
                           data=[4.0, 6.0], shape=[2])
        await job.add_gradient(s1)
        await job.add_gradient(s2)
        result = await job.aggregate_gradients(1)
        assert "fc1" in result
        expected = np.array([3.0, 5.0])
        np.testing.assert_array_almost_equal(result["fc1"], expected)

    def test_create_checkpoint(self):
        job = TrainingJob(TrainingConfig())
        ckpt = job.create_checkpoint(step=100, loss=3.5, metrics={"acc": 0.8})
        assert ckpt.step == 100
        assert ckpt.loss == 3.5

    @pytest.mark.asyncio
    async def test_cleanup_old_gradients(self):
        job = TrainingJob(TrainingConfig())
        for step in range(10):
            s = GradientShard(shard_id=f"s-{step}", job_id=job.job_id, step=step,
                             layer_name="l", data=[float(step)], shape=[1])
            await job.add_gradient(s)
        await job.cleanup_old_gradients(keep_last=3)
        for step in range(7):
            assert step not in job.gradient_buffer
        for step in range(7, 10):
            assert step in job.gradient_buffer


class TestLocalTrainer:
    @pytest.mark.asyncio
    async def test_train_step(self):
        config = TrainingConfig(total_steps=10, num_layers=2, hidden_size=32, num_heads=4,
                               vocab_size=100, intermediate_size=64,
                               checkpoint_interval=1000)
        trainer = LocalTrainer(config)
        metrics = await trainer.train_step()
        assert metrics.step == 1
        assert metrics.loss > 0

    @pytest.mark.asyncio
    async def test_train_loop(self):
        config = TrainingConfig(total_steps=5, num_layers=1, hidden_size=16, num_heads=4,
                               vocab_size=50, intermediate_size=32,
                               checkpoint_interval=1000)
        trainer = LocalTrainer(config)
        job = await trainer.train()
        assert job.status == TrainingStatus.COMPLETED
        assert job.current_step == 5

    @pytest.mark.asyncio
    async def test_stop_training(self):
        config = TrainingConfig(total_steps=10000, num_layers=1, hidden_size=16, num_heads=4,
                               vocab_size=50, intermediate_size=32,
                               checkpoint_interval=1000)
        trainer = LocalTrainer(config)
        trainer.job.started_at = time.time()
        trainer._running = True

        async def stop_after():
            await asyncio.sleep(0.05)
            trainer.stop()

        task = asyncio.create_task(trainer.train())
        asyncio.create_task(stop_after())
        await task
        assert trainer.job.status in (TrainingStatus.CANCELLED, TrainingStatus.COMPLETED)

    def test_get_set_model_state(self):
        config = TrainingConfig(total_steps=1, num_layers=1, hidden_size=16, num_heads=4,
                               vocab_size=50, intermediate_size=32)
        trainer = LocalTrainer(config)
        trainer._init_model()
        state = trainer.get_model_state()
        assert len(state) > 0
        trainer2 = LocalTrainer(config)
        trainer2.load_model_state(state)
        assert len(trainer2._model_params) > 0


class TestGradientCompressor:
    def test_topk_compress(self):
        grad = np.random.randn(1000).astype(np.float32)
        compressed = GradientCompressor.compress(grad, method="topk", ratio=0.01)
        assert compressed["method"] == "topk"
        assert len(compressed["indices"]) == 10
        assert len(compressed["values"]) == 10

    def test_topk_roundtrip(self):
        grad = np.random.randn(100).astype(np.float32)
        compressed = GradientCompressor.compress(grad, method="topk", ratio=0.1)
        decompressed = GradientCompressor.decompress(compressed)
        assert decompressed.shape == grad.shape
        nonzero = np.count_nonzero(decompressed)
        assert nonzero == 10

    def test_quantize_roundtrip(self):
        grad = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        compressed = GradientCompressor.compress(grad, method="quantize")
        decompressed = GradientCompressor.decompress(compressed)
        assert decompressed.shape == grad.shape
        np.testing.assert_array_almost_equal(decompressed, grad, decimal=1)


class TestLearningRateScheduler:
    def test_warmup(self):
        sched = LearningRateScheduler(base_lr=3e-4, warmup_steps=100, total_steps=1000)
        lr_0 = sched.get_lr(0)
        lr_50 = sched.get_lr(50)
        lr_100 = sched.get_lr(100)
        assert lr_0 < lr_50
        assert abs(lr_100 - 3e-4) < 1e-6

    def test_cosine_decay(self):
        sched = LearningRateScheduler(base_lr=1e-3, warmup_steps=0, total_steps=1000, schedule="cosine")
        lr_start = sched.get_lr(0)
        lr_end = sched.get_lr(999)
        assert lr_end < lr_start

    def test_linear_decay(self):
        sched = LearningRateScheduler(base_lr=1e-3, warmup_steps=0, total_steps=1000, schedule="linear")
        lr_end = sched.get_lr(999)
        assert lr_end < 1e-3


class TestShardScheduler:
    def test_assign_shards_single_node(self):
        config = TrainingConfig(num_layers=12, num_shards=1)
        job = TrainingJob(config)
        sched = ShardScheduler()
        nodes = [{"node_id": "n1", "gpu_count": 1}]
        assignments = sched.assign_shards(job, nodes)
        assert len(assignments) == 1

    def test_assign_shards_multi_node(self):
        config = TrainingConfig(num_layers=12, num_shards=3)
        job = TrainingJob(config)
        sched = ShardScheduler()
        nodes = [
            {"node_id": "n1", "gpu_count": 1},
            {"node_id": "n2", "gpu_count": 1},
            {"node_id": "n3", "gpu_count": 0},
        ]
        assignments = sched.assign_shards(job, nodes)
        assert len(assignments) == 3
        all_layers = []
        for sa in assignments.values():
            all_layers.extend(sa.layers)
        assert len(all_layers) == 12

    def test_reassign_failed_shard(self):
        config = TrainingConfig(num_layers=12, num_shards=2)
        job = TrainingJob(config)
        sched = ShardScheduler()
        nodes = [{"node_id": "n1", "gpu_count": 1}, {"node_id": "n2", "gpu_count": 1}]
        assignments = sched.assign_shards(job, nodes)
        sid = list(assignments.keys())[0]
        result = sched.reassign_failed_shard(job, sid, "n3")
        assert result is not None
        assert result.node_id == "n3"


class TestCheckpointManager:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = CheckpointManager(td)
            config = TrainingConfig(job_id="test-1", num_layers=1, hidden_size=16, num_heads=4,
                                   vocab_size=50, intermediate_size=32)
            job = TrainingJob(config)
            state = {"w": np.random.randn(16, 16).astype(np.float32)}
            ckpt = mgr.save_checkpoint(job, step=1, model_state=state, loss=5.0)
            assert ckpt.step == 1
            result = mgr.load_checkpoint("test-1", step=1)
            assert result is not None
            data, meta = result
            assert "w" in data

    def test_list_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = CheckpointManager(td)
            config = TrainingConfig(job_id="test-2", num_layers=1, hidden_size=16, num_heads=4,
                                   vocab_size=50, intermediate_size=32)
            job = TrainingJob(config)
            state = {"w": np.zeros(1)}
            mgr.save_checkpoint(job, step=100, model_state=state, loss=5.0)
            mgr.save_checkpoint(job, step=200, model_state=state, loss=4.0)
            ls = mgr.list_checkpoints("test-2")
            assert len(ls) == 2

    def test_prune_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = CheckpointManager(td)
            config = TrainingConfig(job_id="test-3", num_layers=1, hidden_size=16, num_heads=4,
                                   vocab_size=50, intermediate_size=32)
            job = TrainingJob(config)
            state = {"w": np.zeros(1)}
            for step in [100, 200, 300, 400, 500]:
                mgr.save_checkpoint(job, step=step, model_state=state, loss=5.0)
            mgr.prune_checkpoints("test-3", keep=2)
            ls = mgr.list_checkpoints("test-3")
            assert len(ls) == 2


# ── Voting System ──

class TestResourcePledge:
    def test_compute_score(self):
        p = ResourcePledge(user_id="u1", cpu_cores=8, gpu_count=1, ram_gb=32, gpu_vram_mb=[8000])
        assert p.compute_score > 0

    def test_summary(self):
        p = ResourcePledge(user_id="u1", cpu_cores=4, gpu_count=2, ram_gb=16)
        s = p.summary
        assert "4 CPU cores" in s
        assert "2 GPU" in s

    def test_empty_pledge(self):
        p = ResourcePledge(user_id="u1", time_hours=0)
        assert p.summary == "No resources"


class TestVotingEngine:
    @pytest.fixture
    def engine(self):
        return VotingEngine()

    def test_create_model_proposal(self, engine):
        model = UserModelProposal(model_name="gpt-neo-2.7b", proposer_id="u1")
        p = engine.create_model_proposal(model, "u1")
        assert p.proposal_id
        assert p.status == ProposalStatus.ACTIVE

    def test_cast_vote(self, engine):
        model = UserModelProposal(model_name="gpt2", proposer_id="u1")
        p = engine.create_model_proposal(model, "u1")
        vote = engine.cast_vote(p.proposal_id, "u2", "for")
        assert vote is not None
        assert vote.choice == "for"

    def test_vote_weight_by_resource(self, engine):
        engine.vote_weight_type = VoteWeight.BY_RESOURCE
        pledge = ResourcePledge(user_id="u2", cpu_cores=8, gpu_count=1, ram_gb=32)
        engine.pledges["u2"] = pledge
        model = UserModelProposal(model_name="gpt2", proposer_id="u1")
        p = engine.create_model_proposal(model, "u1")
        vote = engine.cast_vote(p.proposal_id, "u2", "for")
        assert vote.weight > 1.0

    def test_cannot_vote_on_nonexistent(self, engine):
        vote = engine.cast_vote("nonexistent", "u1", "for")
        assert vote is None

    def test_change_vote(self, engine):
        model = UserModelProposal(model_name="gpt2", proposer_id="u1")
        p = engine.create_model_proposal(model, "u1")
        engine.cast_vote(p.proposal_id, "u2", "for")
        engine.cast_vote(p.proposal_id, "u2", "against")
        assert p.votes_for == 0
        assert p.votes_against == 1

    def test_list_proposals(self, engine):
        for i in range(3):
            m = UserModelProposal(model_name=f"model-{i}", proposer_id="u1")
            engine.create_model_proposal(m, "u1")
        props = engine.list_proposals()
        assert len(props) == 3

    def test_cancel_proposal(self, engine):
        m = UserModelProposal(model_name="test", proposer_id="u1")
        p = engine.create_model_proposal(m, "u1")
        result = engine.cancel_proposal(p.proposal_id, "u1")
        assert result is True
        assert p.status == ProposalStatus.CANCELLED

    def test_resource_pledge_proposal(self, engine):
        pledge = ResourcePledge(user_id="u1", cpu_cores=4, gpu_count=1, ram_gb=16)
        p = engine.create_resource_pledge(pledge)
        assert p.vote_type == VoteType.RESOURCE_PLEDGE

    def test_leaderboard(self, engine):
        p1 = ResourcePledge(user_id="u1", cpu_cores=8, gpu_count=2, ram_gb=64)
        p2 = ResourcePledge(user_id="u2", cpu_cores=4, gpu_count=1, ram_gb=32)
        engine.pledges["u1"] = p1
        engine.pledges["u2"] = p2
        lb = engine.get_leaderboard()
        assert len(lb) == 2
        assert lb[0]["rank"] == 1
        assert lb[0]["user_id"] == "u1"

    def test_cluster_resources(self, engine):
        engine.pledges["u1"] = ResourcePledge(user_id="u1", cpu_cores=8, gpu_count=2, ram_gb=64)
        engine.pledges["u2"] = ResourcePledge(user_id="u2", cpu_cores=4, gpu_count=1, ram_gb=32)
        cr = engine.get_cluster_resources()
        assert cr["total_cpu_cores"] == 12
        assert cr["total_gpu_count"] == 3
        assert cr["num_contributors"] == 2


# ── Groups ──

class TestGroupManager:
    @pytest.fixture
    def gm(self):
        return GroupManager()

    def test_create_group(self, gm):
        g = gm.create_group("test-team", "owner1")
        assert g.group_id
        assert g.name == "test-team"
        assert g.owner_id == "owner1"

    def test_join_group(self, gm):
        g = gm.create_group("test", "owner1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(require_approval=False))
        ok, msg = gm.join_group(g.group_id, "user1")
        assert ok

    def test_join_private_requires_invite(self, gm):
        g = gm.create_group("private", "owner1", visibility=GroupVisibility.PRIVATE)
        ok, msg = gm.join_group(g.group_id, "user1")
        assert not ok

    def test_join_with_invite(self, gm):
        g = gm.create_group("invite-only", "owner1", visibility=GroupVisibility.PRIVATE)
        code = gm.create_invite(g.group_id, "owner1")
        assert code is not None
        ok, msg = gm.join_group(g.group_id, "user1", invite_code=code)
        assert ok

    def test_join_with_pledge(self, gm):
        g = gm.create_group("pledge-group", "owner1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(require_approval=False, min_resource_score=5.0))
        pledge = ResourcePledge(user_id="user1", cpu_cores=2, gpu_count=1, ram_gb=16)
        ok, msg = gm.join_group(g.group_id, "user1", pledge=pledge)
        assert ok

    def test_leave_group(self, gm):
        g = gm.create_group("test", "owner1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(require_approval=False))
        gm.join_group(g.group_id, "user1")
        result = gm.leave_group(g.group_id, "user1")
        assert result

    def test_owner_cannot_leave(self, gm):
        g = gm.create_group("test", "owner1")
        result = gm.leave_group(g.group_id, "owner1")
        assert not result

    def test_create_invite(self, gm):
        g = gm.create_group("test", "owner1")
        code = gm.create_invite(g.group_id, "owner1")
        assert code is not None

    def test_non_admin_cannot_invite(self, gm):
        g = gm.create_group("test", "owner1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(require_approval=False))
        gm.join_group(g.group_id, "user1")
        code = gm.create_invite(g.group_id, "user1")
        assert code is None

    def test_remove_member(self, gm):
        g = gm.create_group("test", "owner1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(require_approval=False))
        gm.join_group(g.group_id, "user1")
        ok, msg = gm.remove_member(g.group_id, "owner1", "user1")
        assert ok

    def test_set_member_role(self, gm):
        g = gm.create_group("test", "owner1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(require_approval=False))
        gm.join_group(g.group_id, "user1")
        ok, msg = gm.set_member_role(g.group_id, "owner1", "user1", MemberRole.ADMIN)
        assert ok

    def test_list_groups(self, gm):
        gm.create_group("team1", "o1", visibility=GroupVisibility.PUBLIC)
        gm.create_group("team2", "o2", visibility=GroupVisibility.PRIVATE)
        gm.create_group("team3", "o3", visibility=GroupVisibility.SECRET)
        groups_all = gm.list_groups()
        assert len(groups_all) >= 2
        groups_no_secret = gm.list_groups(visibility=GroupVisibility.PUBLIC)
        assert any(g["visibility"] == "public" for g in groups_no_secret)

    def test_get_group_resources(self, gm):
        g = gm.create_group("test", "owner1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(require_approval=False))
        pledge = ResourcePledge(user_id="user1", cpu_cores=8, gpu_count=2, ram_gb=32)
        gm.join_group(g.group_id, "user1", pledge=pledge)
        res = gm.get_group_resources(g.group_id)
        assert res["pledged_cpu_cores"] == 8
        assert res["pledged_gpu_count"] == 2

    def test_can_start_training(self, gm):
        g = gm.create_group("test", "owner1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(require_approval=False))
        pledge = ResourcePledge(user_id="owner1", cpu_cores=4, gpu_count=1, ram_gb=16)
        gm.groups[g.group_id].members["owner1"].pledge = pledge
        gm.groups[g.group_id].members["owner1"].status = "active"
        gm.voting.pledges["owner1"] = pledge
        ok, msg = gm.can_start_training(g.group_id, "owner1")
        assert ok

    def test_validate_resource_access(self, gm):
        g = gm.create_group("test", "owner1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(require_approval=False))
        pledge = ResourcePledge(user_id="owner1", cpu_cores=8, gpu_count=2, ram_gb=32)
        gm.groups[g.group_id].members["owner1"].pledge = pledge
        gm.groups[g.group_id].members["owner1"].status = "active"
        ok, msg = gm.validate_resource_access(g.group_id, "owner1", required_gpu=1)
        assert ok
        ok2, msg2 = gm.validate_resource_access(g.group_id, "owner1", required_gpu=5)
        assert not ok2

    def test_propose_training(self, gm):
        g = gm.create_group("test", "owner1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(require_approval=False))
        proposal, pid = gm.propose_training(g.group_id, "owner1",
                                              {"model_name": "gpt2", "total_steps": 1000})
        assert proposal is not None
        assert pid


# ── GitHub Integration ──

class TestGitHubIntegration:
    def test_verify_webhook_no_secret(self):
        gh = GitHubIntegration(GitHubConfig())
        assert gh.verify_webhook(b"test", "") is False

    def test_verify_webhook_with_secret(self):
        import hmac, hashlib
        secret = "my-secret"
        gh = GitHubIntegration(GitHubConfig(webhook_secret=secret))
        payload = b'{"test": true}'
        sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        assert gh.verify_webhook(payload, sig) is True
        assert gh.verify_webhook(payload, "wrong") is False

    def test_parse_push_event(self):
        gh = GitHubIntegration(GitHubConfig(branch="main", trigger_paths=["model/"]))
        payload = {
            "ref": "refs/heads/main",
            "repository": {"full_name": "user/repo"},
            "sender": {"login": "user"},
            "commits": [{
                "id": "abc123",
                "message": "update model",
                "author": {"name": "dev"},
                "added": ["model/config.yaml"],
                "modified": [],
                "removed": [],
            }],
        }
        event = gh.parse_webhook_event({"X-GitHub-Event": "push"}, payload)
        assert event.event_type == "push"
        assert event.should_trigger is True
        assert event.branch == "main"

    def test_push_wrong_branch_no_trigger(self):
        gh = GitHubIntegration(GitHubConfig(branch="main"))
        payload = {
            "ref": "refs/heads/dev",
            "repository": {"full_name": "user/repo"},
            "commits": [],
        }
        event = gh.parse_webhook_event({"X-GitHub-Event": "push"}, payload)
        assert event.should_trigger is False

    def test_no_trigger_paths(self):
        gh = GitHubIntegration(GitHubConfig(branch="main", trigger_paths=["model/"]))
        payload = {
            "ref": "refs/heads/main",
            "repository": {"full_name": "user/repo"},
            "commits": [{"id": "a", "message": "fix typo", "author": {"name": "d"},
                        "added": ["README.md"], "modified": [], "removed": []}],
        }
        event = gh.parse_webhook_event({"X-GitHub-Event": "push"}, payload)
        assert event.should_trigger is False


# ── Job Scheduler ──

class TestJobScheduler:
    @pytest.fixture
    def sched(self):
        return JobScheduler()

    def test_register_node(self, sched):
        n = NodeResources(node_id="n1", cpu_cores=8, cpu_available=8, gpu_count=1, gpu_available=1, ram_gb=32, ram_available_gb=32)
        sched.register_node(n)
        assert "n1" in sched.nodes

    def test_submit_job(self, sched):
        n = NodeResources(node_id="n1", cpu_cores=8, cpu_available=8, ram_gb=32, ram_available_gb=32)
        sched.register_node(n)
        jid = sched.submit_job(JobRequirements(min_cpu_cores=2, min_ram_gb=4), name="test")
        assert jid

    def test_schedule_assigns(self, sched):
        n = NodeResources(node_id="n1", cpu_cores=8, cpu_available=8, gpu_count=0, gpu_available=0, ram_gb=32, ram_available_gb=32)
        sched.register_node(n)
        sched.submit_job(JobRequirements(min_cpu_cores=2, min_ram_gb=4))
        assignments = sched.schedule()
        assert len(assignments) == 1

    def test_schedule_gpu_job(self, sched):
        n = NodeResources(node_id="n1", cpu_cores=8, cpu_available=8, gpu_count=1, gpu_available=1, ram_gb=32, ram_available_gb=32)
        sched.register_node(n)
        sched.submit_job(JobRequirements(min_gpu_count=1, min_ram_gb=8))
        assignments = sched.schedule()
        assert len(assignments) == 1

    def test_no_nodes_available(self, sched):
        sched.submit_job(JobRequirements(min_gpu_count=4))
        assignments = sched.schedule()
        assert len(assignments) == 0
        assert len(sched.queue) == 1

    def test_complete_job(self, sched):
        n = NodeResources(node_id="n1", cpu_cores=8, cpu_available=8, ram_gb=32, ram_available_gb=32)
        sched.register_node(n)
        jid = sched.submit_job(JobRequirements(min_cpu_cores=1))
        sched.schedule()
        sched.complete_job(jid)
        assert jid in sched.completed_jobs

    def test_get_queue_status(self, sched):
        n = NodeResources(node_id="n1", cpu_cores=8, cpu_available=8, ram_gb=32, ram_available_gb=32)
        sched.register_node(n)
        status = sched.get_queue_status()
        assert status["nodes_registered"] == 1

    def test_group_filtering(self, sched):
        n1 = NodeResources(node_id="n1", cpu_cores=8, cpu_available=8, ram_gb=32, ram_available_gb=32, group_id="g1")
        n2 = NodeResources(node_id="n2", cpu_cores=8, cpu_available=8, ram_gb=32, ram_available_gb=32, group_id="g2")
        sched.register_node(n1)
        sched.register_node(n2)
        sched.submit_job(JobRequirements(min_cpu_cores=1, group_id="g1"))
        assignments = sched.schedule()
        assert len(assignments) == 1
        assert "n1" in assignments[0][1]


# ── Crypto ──

class TestNodeIdentity:
    def test_generate(self):
        ident = NodeIdentity.generate("test-node")
        assert ident.node_id == "test-node"
        assert ident.verification_key is not None
        assert ident.dh_public_key is not None

    def test_sign_verify(self):
        ident = NodeIdentity.generate("test")
        data = b"hello world"
        sig = ident.sign(data)
        assert ident.verify(data, sig)

    def test_verify_wrong_data(self):
        ident = NodeIdentity.generate("test")
        sig = ident.sign(b"correct")
        assert not ident.verify(b"wrong", sig)

    def test_export_import(self):
        ident = NodeIdentity.generate("test")
        d = ident.to_dict()
        priv = ident.export_private()
        ident2 = NodeIdentity.from_dict(d, priv)
        assert ident2.node_id == "test"
        assert ident2.signing_key is not None


class TestGroupKey:
    def test_derive(self):
        gk = derive_group_key("group-1", "passphrase123")
        assert gk.shared_secret is not None
        assert len(gk.shared_secret) == 32

    def test_encrypt_decrypt(self):
        gk = derive_group_key("group-1")
        plaintext = b"secret training data"
        encrypted = gk.encrypt(plaintext)
        decrypted = gk.decrypt(encrypted)
        assert decrypted == plaintext

    def test_rotate(self):
        gk = derive_group_key("group-1")
        gk2 = gk.rotate()
        assert gk2.key_version == 2
        assert gk2.shared_secret != gk.shared_secret

    def test_encrypt_with_aad(self):
        gk = derive_group_key("group-1")
        plaintext = b"data"
        aad = b"authenticated"
        encrypted = gk.encrypt(plaintext, aad)
        decrypted = gk.decrypt(encrypted, aad)
        assert decrypted == plaintext


class TestPasswordHashing:
    def test_hash_and_verify(self):
        h, salt = hash_password("mypassword")
        assert verify_password("mypassword", h, salt)

    def test_wrong_password(self):
        h, salt = hash_password("mypassword")
        assert not verify_password("wrongpassword", h, salt)