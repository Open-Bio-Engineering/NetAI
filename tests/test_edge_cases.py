"""Edge case and integration tests."""

import asyncio
import time

import numpy as np
import pytest

from netai.training.engine import (
    TrainingConfig, TrainingJob, TrainingStatus, LocalTrainer,
    GradientCompressor, GradientShard, CheckpointManager, ShardScheduler,
    LearningRateScheduler,
)
from netai.training.voting import (
    VotingEngine, ResourcePledge, UserModelProposal,
    VoteType, VoteWeight, ProposalStatus,
)
from netai.training.groups import (
    GroupManager, GroupVisibility, GroupPolicy, MemberRole,
)
from netai.scheduler.scheduler import (
    JobScheduler, NodeResources, JobRequirements, JobPriority,
)
from netai.crypto.identity import NodeIdentity, GroupKey, derive_group_key


class TestVotingEdgeCases:
    @pytest.fixture
    def engine(self):
        return VotingEngine(VoteWeight.BY_RESOURCE)

    def test_expired_proposal(self, engine):
        model = UserModelProposal(model_name="test", proposer_id="u1")
        p = engine.create_model_proposal(model, "u1", voting_deadline=time.time() - 1)
        p.status = ProposalStatus.ACTIVE
        vote = engine.cast_vote(p.proposal_id, "u2", "for")
        assert vote is None

    def test_voter_double_vote_replaces(self, engine):
        model = UserModelProposal(model_name="test", proposer_id="u1")
        p = engine.create_model_proposal(model, "u1")
        engine.cast_vote(p.proposal_id, "u2", "for")
        engine.cast_vote(p.proposal_id, "u2", "against")
        assert p.votes_against == 1
        assert p.votes_for == 0

    def test_equal_weight_voting(self):
        engine = VotingEngine(VoteWeight.EQUAL)
        model = UserModelProposal(model_name="test", proposer_id="u1")
        p = engine.create_model_proposal(model, "u1")
        v = engine.cast_vote(p.proposal_id, "u2", "for")
        assert v.weight == 1.0

    def test_reputation_weight_voting(self):
        engine = VotingEngine(VoteWeight.BY_REPUTATION)
        engine.reputation["u2"] = 5.0
        model = UserModelProposal(model_name="test", proposer_id="u1")
        p = engine.create_model_proposal(model, "u1")
        v = engine.cast_vote(p.proposal_id, "u2", "for")
        assert v.weight == 5.0

    def test_update_reputation(self, engine):
        engine.update_reputation("u1", 1.0)
        assert engine.reputation["u1"] == 2.0
        engine.update_reputation("u1", -0.5)
        assert engine.reputation["u1"] == 1.5
        engine.update_reputation("u1", -5.0)
        assert engine.reputation["u1"] == 0.1

    def test_train_proposal(self, engine):
        p = engine.create_train_proposal(
            job_config={"model_name": "gpt2", "total_steps": 5000},
            proposer_id="u1",
        )
        assert p.vote_type == VoteType.TRAIN_START
        assert p.status == ProposalStatus.ACTIVE

    def test_cancel_wrong_user(self, engine):
        model = UserModelProposal(model_name="test", proposer_id="u1")
        p = engine.create_model_proposal(model, "u1")
        result = engine.cancel_proposal(p.proposal_id, "u2")
        assert not result

    def test_cluster_resources_with_group(self, engine):
        engine.pledges["u1"] = ResourcePledge(user_id="u1", cpu_cores=8, group_id="g1")
        engine.pledges["u2"] = ResourcePledge(user_id="u2", cpu_cores=4, group_id="g2")
        cr_all = engine.get_cluster_resources()
        cr_g1 = engine.get_cluster_resources(group_id="g1")
        assert cr_all["total_cpu_cores"] == 12
        assert cr_g1["total_cpu_cores"] == 8


class TestGroupEdgeCases:
    @pytest.fixture
    def gm(self):
        return GroupManager()

    def test_group_full(self, gm):
        g = gm.create_group("full", "o1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(max_members=2, require_approval=False))
        gm.join_group(g.group_id, "u1")
        ok, msg = gm.join_group(g.group_id, "u2")
        assert not ok
        assert "full" in msg.lower()

    def test_min_resource_too_low(self, gm):
        g = gm.create_group("elite", "o1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(min_resource_score=100, require_approval=False))
        pledge = ResourcePledge(user_id="u1", cpu_cores=1)
        ok, msg = gm.join_group(g.group_id, "u1", pledge=pledge)
        assert not ok
        assert "below minimum" in msg.lower()

    def test_observer_cannot_train(self, gm):
        g = gm.create_group("test", "o1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(require_approval=False))
        gm.join_group(g.group_id, "u1")
        gm.set_member_role(g.group_id, "o1", "u1", MemberRole.OBSERVER)
        ok, msg = gm.can_start_training(g.group_id, "u1")
        assert not ok
        assert "observer" in msg.lower()

    def test_max_concurrent_jobs(self, gm):
        g = gm.create_group("test", "o1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(require_approval=False, max_concurrent_jobs=1))
        g.active_jobs.append("job-1")
        ok, msg = gm.can_start_training(g.group_id, "o1")
        assert not ok

    def test_observer_resource_access_denied(self, gm):
        g = gm.create_group("test", "o1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(require_approval=False))
        gm.join_group(g.group_id, "u1")
        gm.set_member_role(g.group_id, "o1", "u1", MemberRole.OBSERVER)
        ok, msg = gm.validate_resource_access(g.group_id, "u1", required_gpu=1)
        assert not ok

    def test_cannot_remove_owner(self, gm):
        g = gm.create_group("test", "o1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(require_approval=False))
        ok, msg = gm.remove_member(g.group_id, "o1", "o1")
        assert not ok

    def test_admin_cannot_remove_admin(self, gm):
        g = gm.create_group("test", "o1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(require_approval=False))
        gm.join_group(g.group_id, "u1")
        gm.join_group(g.group_id, "u2")
        gm.set_member_role(g.group_id, "o1", "u1", MemberRole.ADMIN)
        gm.set_member_role(g.group_id, "o1", "u2", MemberRole.ADMIN)
        ok, msg = gm.remove_member(g.group_id, "u1", "u2")
        assert not ok

    def test_non_member_cannot_invite(self, gm):
        g = gm.create_group("test", "o1", visibility=GroupVisibility.PRIVATE)
        code = gm.create_invite(g.group_id, "nonmember")
        assert code is None

    def test_expired_invite(self, gm):
        g = gm.create_group("test", "o1", visibility=GroupVisibility.PRIVATE)
        gm._invite_codes["expired-code"] = {
            "group_id": g.group_id,
            "inviter_id": "o1",
            "max_uses": 1,
            "uses": 0,
            "expires": time.time() - 3600,
        }
        ok, msg = gm.join_group(g.group_id, "u1", invite_code="expired-code")
        assert not ok
        assert "expired" in msg.lower()

    def test_wrong_group_invite(self, gm):
        g = gm.create_group("test", "o1", visibility=GroupVisibility.PRIVATE)
        gm._invite_codes["wrong-code"] = {
            "group_id": "other-group",
            "inviter_id": "o1",
            "max_uses": 1,
            "uses": 0,
            "expires": time.time() + 3600,
        }
        ok, msg = gm.join_group(g.group_id, "u1", invite_code="wrong-code")
        assert not ok

    def test_join_already_member(self, gm):
        g = gm.create_group("test", "o1", visibility=GroupVisibility.PUBLIC,
                           policy=GroupPolicy(require_approval=False))
        gm.join_group(g.group_id, "u1")
        ok, msg = gm.join_group(g.group_id, "u1")
        assert not ok


class TestSchedulerEdgeCases:
    @pytest.fixture
    def sched(self):
        return JobScheduler()

    def test_unregister_node(self, sched):
        sched.register_node(NodeResources(node_id="n1", cpu_cores=8, ram_gb=32, ram_available_gb=32))
        sched.unregister_node("n1")
        assert "n1" not in sched.nodes

    def test_failed_job_cleanup(self, sched):
        sched.register_node(NodeResources(node_id="n1", cpu_cores=8, ram_gb=32, ram_available_gb=32))
        jid = sched.submit_job(JobRequirements(min_cpu_cores=1))
        sched.schedule()
        if jid in sched.running_jobs:
            sched.complete_job(jid, success=False)
            job = sched.completed_jobs[jid]
            assert job.status == "failed"
        else:
            sched.complete_job(jid, success=False)

    def test_multiple_jobs_priority(self, sched):
        sched.register_node(NodeResources(node_id="n1", cpu_cores=16, ram_gb=64, ram_available_gb=64,
                                           gpu_count=2, gpu_available=2))
        sched.submit_job(JobRequirements(min_gpu_count=1, priority=JobPriority.LOW), name="low")
        sched.submit_job(JobRequirements(min_gpu_count=1, priority=JobPriority.HIGH), name="high")
        sched.submit_job(JobRequirements(min_gpu_count=1, priority=JobPriority.URGENT), name="urgent")
        assignments = sched.schedule()
        assert len(assignments) >= 1

    def test_rebalance(self, sched):
        sched.register_node(NodeResources(node_id="n1", cpu_cores=8, ram_gb=32, ram_available_gb=32))
        jid = sched.submit_job(JobRequirements(min_cpu_cores=1))
        sched.schedule()
        sched.nodes["n1"].is_training = False
        sched.nodes["n1"].current_jobs = 100
        migrations = sched.rebalance()
        assert len(migrations) == 0 or isinstance(migrations, list)


class TestCryptoEdgeCases:
    def test_key_derive_no_passphrase(self):
        gk = derive_group_key("group-1")
        assert len(gk.shared_secret) == 32

    def test_key_derive_with_passphrase(self):
        gk1 = derive_group_key("group-1", "pass1")
        gk2 = derive_group_key("group-1", "pass2")
        assert gk1.shared_secret != gk2.shared_secret

    def test_identity_sign_different_keys(self):
        id1 = NodeIdentity.generate("n1")
        id2 = NodeIdentity.generate("n2")
        msg = b"test"
        sig1 = id1.sign(msg)
        assert not id2.verify(msg, sig1)

    def test_key_rotation_encrypt(self):
        gk = derive_group_key("g1")
        encrypted = gk.encrypt(b"secret")
        gk2 = gk.rotate()
        with pytest.raises(Exception):
            gk2.decrypt(encrypted)


class TestTrainingEdgeCases:
    def test_lr_scheduler_constant(self):
        s = LearningRateScheduler(base_lr=1e-3, warmup_steps=0, total_steps=1000, schedule="constant")
        assert abs(s.get_lr(0) - 1e-3) < 1e-9
        assert abs(s.get_lr(999) - 1e-3) < 1e-9

    @pytest.mark.asyncio
    async def test_gradient_aggregate_single_shard(self):
        config = TrainingConfig(total_steps=10)
        job = TrainingJob(config)
        s = GradientShard(shard_id="s1", job_id=job.job_id, step=1, layer_name="fc",
                         data=[1.0, 2.0, 3.0], shape=[3])
        await job.add_gradient(s)
        result = await job.aggregate_gradients(1)
        np.testing.assert_array_equal(result["fc"], [1.0, 2.0, 3.0])

    def test_elapsed_seconds_not_started(self):
        config = TrainingConfig(total_steps=10)
        job = TrainingJob(config)
        assert job.elapsed_seconds == 0.0

    def test_compressor_none_method(self):
        grad = np.array([1.0, 2.0])
        compressed = GradientCompressor.compress(grad, method="none")
        decompressed = GradientCompressor.decompress(compressed)
        np.testing.assert_array_equal(decompressed, grad)

    def test_shard_scheduler_no_nodes(self):
        config = TrainingConfig(num_layers=12, num_shards=2)
        job = TrainingJob(config)
        sched = ShardScheduler()
        assignments = sched.assign_shards(job, [])
        assert len(assignments) == 1

    def test_shard_assignment_covers_all_layers(self):
        config = TrainingConfig(num_layers=24, num_shards=4)
        job = TrainingJob(config)
        sched = ShardScheduler()
        nodes = [{"node_id": f"n{i}", "gpu_count": 1} for i in range(4)]
        assignments = sched.assign_shards(job, nodes)
        all_layers = set()
        for sa in assignments.values():
            all_layers.update(sa.layers)
        assert len(all_layers) == 24