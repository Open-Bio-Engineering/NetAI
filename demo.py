"""NetAI Demo - Showcases all features end-to-end."""

import asyncio
import time

from netai.resource.profiler import ResourceProfiler
from netai.training.engine import TrainingConfig, LocalTrainer, GradientCompressor, CheckpointManager
from netai.training.voting import VotingEngine, ResourcePledge, UserModelProposal, VoteWeight
from netai.training.groups import GroupManager, GroupVisibility, GroupPolicy, MemberRole
from netai.github.integration import GitHubIntegration, GitHubConfig
from netai.scheduler.scheduler import JobScheduler, NodeResources, JobRequirements, JobPriority
from netai.crypto.identity import NodeIdentity, derive_group_key
from netai.inference.engine import InferenceEngine, ModelServeConfig, ModelMirror, InferenceRequest
from netai.inference.router import InferenceLoadBalancer, InferenceGateway, InferenceNode, RoutingStrategy
from netai.inference.kv_cache import KVCacheManager, DistributedKVCache
import numpy as np


def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def section(title):
    print(f"\n--- {title} ---\n")


def demo():
    header("NetAI v0.1.0 - Distributed AI Training & Inference Demo")

    # 1. Resource Profiling
    header("1. Resource Profiling")
    profiler = ResourceProfiler()
    profile = profiler.profile()
    print(f"  CPU:      {profile.cpu_available}/{profile.cpu_cores} cores ({profile.cpu_arch}, {profile.cpu_freq_ghz:.2f} GHz)")
    print(f"  GPU:      {profile.gpu_count} ({', '.join(profile.gpu_names) or 'none'})")
    if profile.gpu_vram_mb:
        print(f"  GPU VRAM: {profile.gpu_vram_mb}")
    print(f"  RAM:      {profile.ram_available_gb:.1f}/{profile.ram_total_gb:.1f} GB")
    print(f"  Disk:     {profile.disk_available_gb:.1f}/{profile.disk_total_gb:.1f} GB")
    print(f"  Backends: CUDA={profile.has_cuda} ROCm={profile.has_rocm} Vulkan={profile.has_vulkan}")
    print(f"  PyTorch:  {profile.torch_available} ({profile.torch_gpu_count} GPUs)")
    print(f"  Score:    {profile.training_capacity_score:.1f}")
    print(f"  Summary:  {profile.summary}")

    # 2. Voting & Resource Pledging
    header("2. Voting System & Resource Pledging")
    voting = VotingEngine(VoteWeight.BY_RESOURCE)

    section("Users pledge resources")
    users = [
        ("alice", 8, 2, 64, [12000, 8000]),
        ("bob", 4, 1, 32, [8000]),
        ("charlie", 16, 0, 32, []),
        ("dora", 4, 1, 16, [4000]),
    ]
    for uid, cpu, gpu, ram, vram in users:
        pledge = ResourcePledge(user_id=uid, cpu_cores=cpu, gpu_count=gpu, ram_gb=ram, gpu_vram_mb=vram)
        voting.create_resource_pledge(pledge)
        print(f"  {uid:10s}: {pledge.summary:40s} (score={pledge.compute_score:.1f})")

    section("Cluster resources")
    cr = voting.get_cluster_resources()
    print(f"  Total CPU:  {cr['total_cpu_cores']} cores")
    print(f"  Total GPU:  {cr['total_gpu_count']}")
    print(f"  Total RAM:  {cr['total_ram_gb']:.0f} GB")
    print(f"  Total VRAM: {cr['total_vram_mb']} MB")
    print(f"  Contributors: {cr['num_contributers']}")

    section("Leaderboard")
    lb = voting.get_leaderboard()
    for e in lb:
        print(f"  #{e['rank']} {e['user_id']:10s} score={e['score']:.1f} ({e['summary']})")

    section("Propose model to train")
    model1 = UserModelProposal(
        model_name="llama-3.1-8b-finetune",
        architecture="transformer",
        description="Fine-tune LLaMA 3.1 8B on medical data",
        proposer_id="alice",
        tags=["llm", "medical", "finetune"],
    )
    p1 = voting.create_model_proposal(model1, "alice")
    print(f"  Proposal: {p1.proposal_id} - {p1.title}")

    model2 = UserModelProposal(
        model_name="stable-diffusion-xl-v2",
        architecture="diffusion",
        description="Train SDXL v2 on custom art dataset",
        proposer_id="bob",
        tags=["diffusion", "art", "generative"],
    )
    p2 = voting.create_model_proposal(model2, "bob")
    print(f"  Proposal: {p2.proposal_id} - {p2.title}")

    section("Cast votes (resource-weighted)")
    voting.cast_vote(p1.proposal_id, "alice", "for", reasoning="Medical AI is high priority")
    voting.cast_vote(p1.proposal_id, "bob", "for", reasoning="Good use case")
    voting.cast_vote(p1.proposal_id, "charlie", "against", reasoning="Too resource-intensive")
    voting.cast_vote(p1.proposal_id, "dora", "for")

    voting.cast_vote(p2.proposal_id, "alice", "against")
    voting.cast_vote(p2.proposal_id, "bob", "for")
    voting.cast_vote(p2.proposal_id, "charlie", "against")
    voting.cast_vote(p2.proposal_id, "dora", "for")

    for p in [p1, p2]:
        total = p.weighted_for + p.weighted_against
        for_pct = (p.weighted_for / total * 100) if total > 0 else 50
        print(f"  {p.title}: {p.weighted_for:.1f} for / {p.weighted_against:.1f} against ({for_pct:.0f}% for) -> {p.result.value}")

    # 3. Private Groups
    header("3. Private Group Training")
    gm = GroupManager()

    section("Create groups")
    med_group = gm.create_group(
        "medical-ai-lab", "alice",
        description="Medical AI research consortium",
        visibility=GroupVisibility.PRIVATE,
        passphrase="med2024secure",
    )
    print(f"  Created: {med_group.name} ({med_group.group_id}) [private]")

    open_group = gm.create_group(
        "open-ml-collective", "charlie",
        description="Open ML for everyone",
        visibility=GroupVisibility.PUBLIC,
        policy=GroupPolicy(require_approval=False, max_members=500),
    )
    print(f"  Created: {open_group.name} ({open_group.group_id}) [public]")

    section("Invites & joining")
    invite_code = gm.create_invite(med_group.group_id, "alice")
    print(f"  Alice created invite code: {invite_code}")

    ok, msg = gm.join_group(med_group.group_id, "bob", pledge=ResourcePledge(
        user_id="bob", cpu_cores=4, gpu_count=1, ram_gb=32, gpu_vram_mb=[8000],
    ), invite_code=invite_code)
    print(f"  Bob joins medical group: {msg}")

    ok2, msg2 = gm.join_group(open_group.group_id, "dora", pledge=ResourcePledge(
        user_id="dora", cpu_cores=4, gpu_count=1, ram_gb=16, gpu_vram_mb=[4000],
    ))
    print(f"  Dora joins open group: {msg2}")

    section("Group resources")
    for gid in [med_group.group_id, open_group.group_id]:
        res = gm.get_group_resources(gid)
        print(f"  {res['group_name']}: {res['pledged_cpu_cores']} CPU, {res['pledged_gpu_count']} GPU, "
              f"{res['pledged_ram_gb']:.0f}GB RAM, {res['members']} members")

    section("Propose training within group")
    proposal, pid = gm.propose_training(
        med_group.group_id, "alice",
        {"model_name": "medllama-8b", "total_steps": 50000, "learning_rate": 2e-5},
    )
    print(f"  Alice proposes training in medical group: {pid}")

    ok3, msg3 = gm.can_start_training(med_group.group_id, "alice")
    print(f"  Can start training: {msg3}")

    section("Resource access validation")
    ok, msg = gm.validate_resource_access(med_group.group_id, "alice", required_gpu=1, required_ram_gb=8)
    print(f"  Alice requests 1 GPU + 8GB RAM: {msg}")

    # 4. Distributed Scheduling
    header("4. Job Scheduling")
    sched = JobScheduler()
    for uid, cpu, gpu, ram in [("node-alice", 8, 2, 64), ("node-bob", 4, 1, 32),
                                ("node-charlie", 16, 0, 32), ("node-dora", 4, 1, 16)]:
        sched.register_node(NodeResources(node_id=uid, cpu_cores=cpu, cpu_available=cpu,
                                           gpu_count=gpu, gpu_available=gpu, ram_gb=ram, ram_available_gb=ram))
    print(f"  Registered {len(sched.nodes)} nodes")

    sched.submit_job(JobRequirements(min_gpu_count=1, min_ram_gb=16, priority=JobPriority.HIGH),
                     name="medllama-8b", submitter_id="alice")
    sched.submit_job(JobRequirements(min_gpu_count=2, min_ram_gb=48, priority=JobPriority.NORMAL),
                     name="sdxl-v2", submitter_id="bob")
    sched.submit_job(JobRequirements(min_cpu_cores=4, min_ram_gb=8, priority=JobPriority.LOW),
                     name="sentiment-model", submitter_id="charlie")
    assignments = sched.schedule()
    print(f"  Scheduled {len(assignments)} jobs:")
    for jid, nodes in assignments:
        print(f"    {jid} -> {nodes}")

    status = sched.get_queue_status()
    print(f"\n  Scheduler status: queued={status['queued']}, running={status['running']}, "
          f"completed={status['completed']}")
    print(f"  Total resources: {status['total_cpu']} CPU, {status['total_gpu']} GPU, "
          f"{status['total_ram_gb']:.0f}GB RAM")

    # 5. Local Training
    header("5. Local Training Simulation")
    config = TrainingConfig(
        model_name="demo-transformer",
        total_steps=50,
        num_layers=2,
        hidden_size=64,
        num_heads=4,
        vocab_size=1000,
        intermediate_size=128,
        batch_size=4,
        learning_rate=1e-3,
        warmup_steps=5,
        checkpoint_interval=25,
    )
    trainer = LocalTrainer(config)

    print("  Training for 50 steps...")
    t0 = time.time()
    job = asyncio.run(trainer.train())
    elapsed = time.time() - t0
    m = job.latest_metrics
    print(f"  Status:    {job.status.value}")
    print(f"  Steps:     {job.current_step}")
    print(f"  Loss:      {m.loss:.4f} (best: {job.best_loss:.4f})")
    print(f"  LR:        {m.lr:.6f}")
    print(f"  Grad norm: {m.grad_norm:.4f}")
    print(f"  Time:      {elapsed:.2f}s")
    print(f"  Checkpoints: {len(job.checkpoints)}")

    # 6. Gradient Compression
    header("6. Gradient Compression")
    grad = np.random.randn(10000).astype(np.float32)
    original_size = grad.nbytes

    for method in ["topk", "quantize", "none"]:
        compressed = GradientCompressor.compress(grad, method=method, ratio=0.01)
        decompressed = GradientCompressor.decompress(compressed)
        if method == "topk":
            approx_ok = np.count_nonzero(decompressed) == 100
            print(f"  {method:10s}: original={original_size:,}B -> sparse, ~{100} non-zero values, valid={approx_ok}")
        elif method == "quantize":
            diff = np.abs(grad - decompressed)
            max_err = diff.max()
            print(f"  {method:10s}: original={original_size:,}B -> 8-bit quantized, max_error={max_err:.4f}")
        else:
            print(f"  {method:10s}: original={original_size:,}B (no compression)")

    # 7. Cryptography
    header("7. Security & Cryptography")
    section("Node identity")
    id1 = NodeIdentity.generate("node-alice")
    id2 = NodeIdentity.generate("node-bob")
    msg = b"gradient sync data"
    sig = id1.sign(msg)
    print(f"  Alice signs message: valid={id1.verify(msg, sig)}")
    print(f"  Bob verifies Alice's signature: valid={id1.verify(msg, sig)}")
    print(f"  Tampered message: valid={id1.verify(b'wrong', sig)}")

    section("Group encryption")
    gk = derive_group_key("medical-ai-lab", "secure-passphrase")
    plaintext = b"confidential model weights"
    encrypted = gk.encrypt(plaintext)
    decrypted = gk.decrypt(encrypted)
    print(f"  Encrypt/decrypt roundtrip: {plaintext == decrypted}")
    print(f"  Key version: {gk.key_version}")
    gk2 = gk.rotate()
    print(f"  After rotation: v{gk.key_version} -> v{gk2.key_version}")

    # 8. GitHub Integration
    header("8. GitHub Commit-Triggered Training")
    gh = GitHubIntegration(GitHubConfig(
        repo_url="https://github.com/example/ai-model",
        branch="main",
        trigger_paths=["model/", "config/", "netai.yaml"],
    ))
    event = gh.parse_webhook_event(
        {"X-GitHub-Event": "push"},
        {
            "ref": "refs/heads/main",
            "repository": {"full_name": "example/ai-model"},
            "sender": {"login": "alice"},
            "commits": [{
                "id": "abc123def456",
                "message": "Update model config for v2 training",
                "author": {"name": "alice"},
                "added": ["model/config_v2.yaml"],
                "modified": ["netai.yaml"],
                "removed": [],
            }],
        },
    )
    print(f"  Push event: branch={event.branch}, commits={len(event.commits)}")
    print(f"  Should trigger training: {event.should_trigger}")
    print(f"  Config changed: {event.config_changed}")
    print(f"  Commit: {event.commits[0].sha[:8]} - {event.commits[0].message}")

    # 9. Distributed Inference
    header("9. Distributed Inference & Model Mirroring")

    section("Load model for inference")
    inf_engine = InferenceEngine(node_id="demo-node")
    asyncio.run(inf_engine.start())
    config = ModelServeConfig(model_id="gpt2-small", model_name="gpt2-small", num_shards=2)
    replica = asyncio.run(inf_engine.load_model(config))
    print(f"  Model loaded: {replica.model_id} (status={replica.status.value}, shards={len(replica.shard_ids)})")

    section("Run inference")
    request = InferenceRequest(model_id="gpt2-small", prompt="The future of AI is", max_tokens=64)
    response = asyncio.run(inf_engine.infer(request))
    print(f"  Prompt:     '{request.prompt}'")
    print(f"  Response:    '{response.text}'")
    print(f"  Tokens:     {response.tokens_generated}")
    print(f"  Latency:    {response.latency_ms:.1f}ms")
    print(f"  Throughput: {response.tokens_per_second:.1f} tok/s")

    section("Model mirroring")
    mirror = ModelMirror()
    mirror.register_mirror("gpt2-small", "node-alice")
    mirror.register_mirror("gpt2-small", "node-bob")
    mirror.register_mirror("gpt2-small", "node-charlie")
    print(f"  Mirrors for gpt2-small: {mirror.get_mirror_count('gpt2-small')}")
    nearest = mirror.find_nearest_mirror("gpt2-small", "node-bob", {"node-alice": 10.0, "node-bob": 1.0, "node-charlie": 50.0})
    print(f"  Nearest mirror from node-bob: {nearest}")

    section("Inference load balancer (adaptive routing)")
    lb = InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)
    for name, models, gpu, load in [("node-a", ["gpt2-small"], 1, 10), ("node-b", ["gpt2-small"], 2, 30), ("node-c", ["gpt2-small"], 1, 5)]:
        lb.register_node(InferenceNode(
            node_id=name, endpoint=f"http://{name}:8001",
            status="ready", models_loaded=models, gpu_count=gpu, current_load=load,
            last_heartbeat=time.time(),
        ))
    routed = lb.route_request(InferenceRequest(model_id="gpt2-small", prompt="test"))
    print(f"  Routed to: {routed} (adaptive strategy picks least-loaded + lowest-latency)")
    lb_status = lb.get_status()
    print(f"  Cluster: {lb_status['total_nodes']} nodes, {lb_status['available_nodes']} available")

    section("KV cache management")
    kv = KVCacheManager(max_size_mb=100.0)
    kv_data = [[[0.1 * i for i in range(64)] for _ in range(12)]]
    entry = kv.put("gpt2-small", "Hello world", kv_data)
    print(f"  Cached: {entry.model_id} ({entry.size_mb:.4f}MB, {entry.num_layers} layers)")
    result = kv.get("gpt2-small", "Hello world")
    print(f"  Cache hit: {result is not None} (access_count={result.access_count if result else 0})")
    stats = kv.get_stats()
    print(f"  Cache stats: {stats['entries']} entries, {stats['size_mb']}MB used, hit_rate={stats['hit_rate']}")

    section("Jack-in: unified training + inference")
    print("  POST /api/jack-in with mode='both' lets anyone join for:")
    print("    - Training (pledge CPU/GPU/RAM, vote on models, run jobs)")
    print("    - Inference (serve models, earn reputation, mirror weights)")
    print("    - Or both simultaneously")
    asyncio.run(inf_engine.stop())

    # Summary
    header("Summary")
    print("  Features demonstrated:")
    print("    1. Resource profiling (CPU, GPU, RAM, backends)")
    print("    2. Voting system (model proposals, weighted votes)")
    print("    3. Resource pledging & leaderboard")
    print("    4. Private group training (invite, join, propose)")
    print("    5. Job scheduling (priority, resource-aware)")
    print("    6. Local training with checkpointing")
    print("    7. Gradient compression (topk, quantize)")
    print("    8. Cryptography (identity, group encryption)")
    print("    9. GitHub commit-triggered training")
    print("   10. Distributed inference + model mirroring")
    print("   11. P2P Jack-in API (unified training+inference)")
    print()
    print("  Start the server: cd netai && PYTHONPATH=src python -m uvicorn netai.api.app:create_app --factory --port 8001")
    print("  Or use CLI:       PYTHONPATH=src python -m netai.cli serve --port 8001")
    print()
    print("  Open dashboard:   http://localhost:8001")
    print()
    print("  Jack In API:      POST /api/jack-in  (join training + inference in one call)")
    print("  Inference Load:   POST /api/inference/load  (load a model)")
    print("  Inference Run:    POST /api/inference/run   (generate text)")


if __name__ == "__main__":
    demo()