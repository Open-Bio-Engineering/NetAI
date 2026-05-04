"""FastAPI web server - REST API for distributed training, voting, groups."""

from __future__ import annotations

import asyncio
import json
import logging
import numpy as np
import os
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, field_validator

from netai.p2p.network import P2PNode, PeerInfo, NodeState
from netai.p2p.handshake import HandshakeProtocol, NodeCapabilities, detect_capabilities
from netai.resource.profiler import ResourceProfiler, ResourceProfile, can_run_model, suggest_batch_size
from netai.training.engine import TrainingConfig, TrainingStatus, DeviceType, OptimizerType
from netai.training.coordinator import DistributedTrainingCoordinator
from netai.training.voting import (
    VotingEngine, ResourcePledge, Proposal, Vote, VoteType, VoteWeight,
    ProposalStatus, UserModelProposal,
)
from netai.training.groups import (
    GroupManager, TrainingGroup, GroupVisibility, MemberRole, GroupPolicy,
)
from netai.github.integration import GitHubIntegration, GitHubConfig, WebhookEvent
from netai.scheduler.scheduler import JobScheduler, NodeResources, JobRequirements, JobPriority
from netai.inference.engine import (
    InferenceEngine, InferenceRequest, InferenceResponse, InferenceStatus,
    ModelServeConfig, ModelMirror,
)
from netai.inference.router import InferenceLoadBalancer, InferenceGateway, InferenceNode, RoutingStrategy
from netai.inference.kv_cache import KVCacheManager, DistributedKVCache
from netai.inference.autoloader import AutoLoader, ModelEntry, ModelRegistry, ModelSizeClass
from netai.inference.native_engine import NativeInferenceEngine, TransformerConfig
from netai.inference.pipeline_executor import PipelineExecutor
from netai.inference.downloader import ModelDownloader
from netai.cache.manager import ModelCacheManager, CacheHitRequest, CacheStatsResponse
from netai.inference.tokenizer import get_tokenizer
from netai.inference.compress import quantize_activation, dequantize_activation, ActivationCompressor
from netai.training.engine import GradientSyncServer
from netai.benchmark.runner import (
    ModelBenchmark, BenchmarkConfig, BenchmarkResult,
    InferenceMetrics, MemoryMetrics, StartupMetrics, PipelineMetrics,
)
from netai.security import (
    SecurityMiddleware, AuthDependency, Scope, UserRole, InputValidator,
    GradientIntegrityChecker, ModelProvenance,
)

logger = logging.getLogger(__name__)


class TrainingRequest(BaseModel):
    model_name: str = "gpt2-small"
    model_repo: str = ""
    model_architecture: str = "transformer"
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 50257
    total_steps: int = 10000
    batch_size: int = 8
    learning_rate: float = 3e-4
    optimizer: str = "adamw"
    device_preference: str = "cuda"
    group_id: str = ""
    dataset: str = ""
    seed: int = 42


class VoteRequest(BaseModel):
    proposal_id: str
    voter_id: str
    choice: str
    reasoning: str = ""
    node_id: str = ""


class PledgeRequest(BaseModel):
    user_id: str
    node_id: str = ""
    cpu_cores: int = 0
    gpu_count: int = 0
    ram_gb: float = 0.0
    gpu_vram_mb: list[int] = Field(default_factory=list)
    time_hours: float = 24.0
    group_id: str = ""


class GroupCreateRequest(BaseModel):
    name: str
    owner_id: str
    description: str = ""
    visibility: str = "private"
    passphrase: str = ""
    max_members: int = 100
    require_approval: bool = True
    tags: list[str] = Field(default_factory=list)


class GroupJoinRequest(BaseModel):
    group_id: str
    user_id: str
    node_id: str = ""
    invite_code: str = ""
    cpu_cores: int = 0
    gpu_count: int = 0
    ram_gb: float = 0.0
    gpu_vram_mb: list[int] = Field(default_factory=list)
    time_hours: float = 24.0


class ModelProposalRequest(BaseModel):
    model_name: str
    architecture: str = "transformer"
    description: str = ""
    repo_url: str = ""
    proposer_id: str = ""
    group_id: str = ""
    tags: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)


class WebhookPayload(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)


class InferenceRunRequest(BaseModel):
    model_id: str
    prompt: str = ""
    inputs: list[Any] = Field(default_factory=list)
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    user_id: str = ""
    group_id: str = ""
    priority: int = 0
    timeout_ms: int = 30000


class TokenizeRequest(BaseModel):
    model_id: str = ""
    text: str = ""
    with_special: bool = False


class DecodeRequest(BaseModel):
    model_id: str = ""
    token_ids: list[int] = Field(default_factory=list)
    skip_special_tokens: bool = True


class CompressRequest(BaseModel):
    data: list = Field(default_factory=list)
    shape: list[int] = Field(default_factory=list)
    bits: int = Field(8, ge=4, le=8)
    model_id: str = ""

    model_config = {"arbitrary_types_allowed": True}


class ModelLoadRequest(BaseModel):
    model_id: str = ""
    model_name: str = "gpt2-small"
    version: str = "latest"
    num_replicas: int = 1
    num_shards: int = 1
    device: str = "auto"
    mirror_enabled: bool = True
    max_batch_size: int = 32
    group_id: str = ""


class AutoLoaderLoadRequest(BaseModel):
    force_models: list[str] = Field(default_factory=list)
    votes: dict[str, float] | None = None


class JackInRequest(BaseModel):
    user_id: str
    node_id: str = ""
    mode: str = "both"
    cpu_cores: int = 0
    gpu_count: int = 0
    ram_gb: float = 0.0
    gpu_vram_mb: list[int] = Field(default_factory=list)
    time_hours: float = 24.0
    group_id: str = ""
    models_to_serve: list[str] = Field(default_factory=list)


class AuthTokenRequest(BaseModel):
    user_id: str
    scopes: list[str] = Field(default_factory=lambda: ["read"])
    ttl_hours: float = 24.0


class ApiKeyRequest(BaseModel):
    user_id: str
    name: str = "default"
    scopes: list[str] = Field(default_factory=lambda: ["read", "write"])


class GradientSyncPayload(BaseModel):
    job_id: str
    step: int
    node_id: str = ""
    gradients: dict[str, Any] = {}
    gradient_hash: str = ""

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, v: str) -> str:
        if len(v) > 128 or not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Invalid job_id")
        return v

    @field_validator("step")
    @classmethod
    def validate_step(cls, v: int) -> int:
        if v < 0 or v > 10_000_000:
            raise ValueError("step must be 0-10M")
        return v

    @field_validator("gradients")
    @classmethod
    def validate_gradients_size(cls, v: dict) -> dict:
        if len(v) > 500:
            raise ValueError("Too many gradient layers (max 500)")
        return v


class LoginRequest(BaseModel):
    user_id: str
    password: str


class RegisterRequest(BaseModel):
    user_id: str
    password: str
    role: str = "user"
    scopes: list[str] = Field(default_factory=list)


def create_app(
    p2p_node: P2PNode | None = None,
    coordinator: DistributedTrainingCoordinator | None = None,
    voting_engine: VotingEngine | None = None,
    group_manager: GroupManager | None = None,
    github: GitHubIntegration | None = None,
    scheduler: JobScheduler | None = None,
    inference_gateway: InferenceGateway | None = None,
    security: SecurityMiddleware | None = None,
    benchmark_runner: ModelBenchmark | None = None,
    native_engine: NativeInferenceEngine | None = None,
) -> FastAPI:
    p2p = p2p_node or P2PNode()
    coord = coordinator or DistributedTrainingCoordinator(p2p)
    voting = voting_engine or VotingEngine()
    groups = group_manager or GroupManager()
    gh = github or GitHubIntegration()
    sched = scheduler or JobScheduler()
    inf_engine = InferenceEngine(node_id=p2p.node_id)
    if native_engine is not None:
        inf_engine._native_engine = native_engine
    inf_lb = InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)
    inf_gw = inference_gateway or InferenceGateway(inf_engine, inf_lb)
    grad_sync = GradientSyncServer(node_id=p2p.node_id)
    sec = security or SecurityMiddleware()
    grad_integrity = GradientIntegrityChecker()
    model_prov = ModelProvenance()
    validator = InputValidator()
    model_registry = ModelRegistry()
    model_registry.load_local()
    model_autoloader = AutoLoader(model_registry)
    ENGINE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "netai")
    handshake = HandshakeProtocol(node_id=p2p.node_id, port=p2p.port)
    benchmark = benchmark_runner or ModelBenchmark(
        engine=inf_engine.get_native_engine(),
        cache_dir=ENGINE_CACHE_DIR,
    )

    metrics_data: dict[str, Any] = {"requests": 0, "errors": 0, "start_time": time.time()}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await inf_gw.start()
        yield
        await inf_gw.stop()

    app = FastAPI(title="NetAI", version="1.0.0",
                  description="Distributed AI Training & Inference with P2P Resource Pooling",
                  lifespan=lifespan)
    cors_origins = sec._cors_origins if sec._cors_origins is not None else []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=bool(cors_origins),
        allow_methods=["*"],
        allow_headers=["*"] if cors_origins else [],
    )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return _dashboard_html()

    @app.get("/api/status")
    async def status():
        profiler = ResourceProfiler()
        profile = profiler.profile()
        peers = await p2p.peer_table.get_alive_peers()
        return {
            "node_id": p2p.node_id,
            "state": p2p.state.value,
            "profile": {
                "cpu_cores": profile.cpu_cores, "cpu_available": profile.cpu_available,
                "cpu_arch": profile.cpu_arch, "cpu_freq_ghz": profile.cpu_freq_ghz,
                "gpu_count": profile.gpu_count, "gpu_available": profile.gpu_available,
                "gpu_names": profile.gpu_names, "gpu_vram_mb": profile.gpu_vram_mb,
                "gpu_available_vram_mb": profile.gpu_available_vram_mb,
                "ram_total_gb": profile.ram_total_gb, "ram_available_gb": profile.ram_available_gb,
                "disk_total_gb": profile.disk_total_gb, "disk_available_gb": profile.disk_available_gb,
                "has_cuda": profile.has_cuda, "has_rocm": profile.has_rocm,
                "has_vulkan": profile.has_vulkan, "torch_available": profile.torch_available,
                "torch_gpu_count": profile.torch_gpu_count,
                "capacity_score": profile.training_capacity_score,
                "summary": profile.summary,
            },
            "peer_count": len(peers),
            "jobs": coord.list_jobs(),
            "scheduler": sched.get_queue_status(),
            "groups": len(groups.groups),
            "proposals": len(voting.proposals),
            "pledges": len(voting.pledges),
            "timestamp": time.time(),
        }

    @app.get("/api/resources")
    async def get_resources():
        profiler = ResourceProfiler()
        profile = profiler.profile()
        return {
            "cpu_cores": profile.cpu_cores,
            "cpu_available": profile.cpu_available,
            "gpu_count": profile.gpu_count,
            "gpu_names": profile.gpu_names,
            "gpu_vram_mb": profile.gpu_vram_mb,
            "gpu_available_vram_mb": profile.gpu_available_vram_mb,
            "ram_total_gb": profile.ram_total_gb,
            "ram_available_gb": profile.ram_available_gb,
            "has_cuda": profile.has_cuda,
            "has_rocm": profile.has_rocm,
            "has_vulkan": profile.has_vulkan,
            "torch_available": profile.torch_available,
            "torch_gpu_count": profile.torch_gpu_count,
            "capacity_score": profile.training_capacity_score,
            "summary": profile.summary,
        }

    @app.get("/api/peers")
    async def get_peers():
        peers = await p2p.peer_table.get_alive_peers()
        return {
            "peers": [p.model_dump() for p in peers],
            "count": len(peers),
        }

    @app.post("/api/p2p/handshake")
    async def p2p_handshake_receive(req: Request):
        body = await req.json()
        caps = handshake.receive_handshake(body)
        return {"accepted": True, "node_id": handshake.node_id,
                "peer_count": len(handshake.peer_capabilities),
                "caps": handshake.capabilities.model_dump()}

    @app.get("/api/p2p/capabilities/{node_id}")
    async def p2p_capabilities(node_id: str):
        return handshake.capabilities.model_dump()

    @app.get("/api/p2p/status")
    async def p2p_handshake_status():
        return handshake.get_status()

    @app.get("/api/p2p/ping")
    async def p2p_ping():
        return {"status": "ok", "node_id": handshake.node_id, "uptime_s": handshake.capabilities.uptime_s}

    @app.post("/api/p2p/score")
    async def p2p_scores():
        return {"scores": {nid: s.model_dump() for nid, s in handshake.peer_scores.items()}}

    @app.get("/api/p2p/suggest-role")
    async def p2p_suggest_role():
        return handshake.suggest_pipeline_role()

    @app.get("/api/p2p/best-for-layers")
    async def p2p_best_for_layers(num_layers: int = 4, memory_per_layer_mb: float = 500):
        return {"candidates": handshake.best_node_for_layers(num_layers, memory_per_layer_mb)}

    @app.post("/api/training/submit")
    async def submit_training(req: TrainingRequest, identity=Depends(AuthDependency(sec, required_scope=Scope.TRAIN.value))):
        validator.validate_model_name(req.model_name)
        validator.validate_positive_int(req.total_steps, "total_steps", 1, 10000000)
        validator.validate_positive_int(req.batch_size, "batch_size", 1, 1024)
        validator.validate_device(req.device_preference)
        config = TrainingConfig(
            model_name=req.model_name,
            model_repo=req.model_repo,
            model_architecture=req.model_architecture,
            hidden_size=req.hidden_size,
            num_layers=req.num_layers,
            num_heads=req.num_heads,
            vocab_size=req.vocab_size,
            total_steps=req.total_steps,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            optimizer=OptimizerType(req.optimizer),
            device_preference=DeviceType(req.device_preference),
            group_id=req.group_id,
            dataset=req.dataset,
            seed=req.seed,
        )
        job = await coord.submit_job(config)
        return {"job_id": job.job_id, "status": job.status.value, "config": config.model_dump()}

    @app.post("/api/training/start/{job_id}")
    async def start_training(job_id: str, identity=Depends(AuthDependency(sec, required_scope=Scope.TRAIN.value))):
        job = await coord.start_training(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return {"job_id": job.job_id, "status": job.status.value}

    @app.post("/api/training/stop/{job_id}")
    async def stop_training(job_id: str, identity=Depends(AuthDependency(sec, required_scope=Scope.TRAIN.value))):
        await coord.stop_training(job_id)
        return {"job_id": job_id, "status": "stopped"}

    @app.get("/api/training/status/{job_id}")
    async def training_status(job_id: str):
        status = coord.get_job_status(job_id)
        if not status:
            raise HTTPException(404, "Job not found")
        return status

    @app.get("/api/training/jobs")
    async def list_jobs():
        return {"jobs": coord.list_jobs()}

    @app.get("/api/training/checkpoints/{job_id}")
    async def list_checkpoints(job_id: str):
        return {"checkpoints": coord.checkpoint_mgr.list_checkpoints(job_id)}

    @app.post("/api/vote/propose-model")
    async def propose_model(req: ModelProposalRequest, identity=Depends(AuthDependency(sec, required_scope=Scope.VOTE.value))):
        model = UserModelProposal(
            model_name=req.model_name,
            architecture=req.architecture,
            description=req.description,
            repo_url=req.repo_url,
            proposer_id=req.proposer_id,
            tags=req.tags,
            config=req.config,
        )
        proposal = voting.create_model_proposal(model, req.proposer_id, req.group_id)
        return {"proposal_id": proposal.proposal_id, "status": proposal.status.value}

    @app.post("/api/vote/cast")
    async def cast_vote(req: VoteRequest, identity=Depends(AuthDependency(sec, required_scope=Scope.VOTE.value))):
        vote = voting.cast_vote(
            proposal_id=req.proposal_id,
            voter_id=req.voter_id,
            choice=req.choice,
            node_id=req.node_id,
            reasoning=req.reasoning,
        )
        if not vote:
            raise HTTPException(400, "Vote rejected")
        return {"vote_id": vote.vote_id, "weight": vote.weight}

    @app.get("/api/vote/proposals")
    async def list_proposals(
        status: str | None = None,
        vote_type: str | None = None,
        group_id: str | None = None,
    ):
        ps = ProposalStatus(status) if status else None
        vt = VoteType(vote_type) if vote_type else None
        proposals = voting.list_proposals(ps, vt, group_id)
        return {
            "proposals": [
                {
                    "proposal_id": p.proposal_id,
                    "title": p.title,
                    "status": p.status.value,
                    "vote_type": p.vote_type.value,
                    "votes_for": p.votes_for,
                    "votes_against": p.votes_against,
                    "weighted_for": round(p.weighted_for, 2),
                    "weighted_against": round(p.weighted_against, 2),
                    "proposer_id": p.proposer_id,
                    "group_id": p.group_id,
                    "quorum": p.quorum,
                    "threshold": p.threshold,
                }
                for p in proposals
            ]
        }

    @app.get("/api/vote/proposal/{proposal_id}")
    async def get_proposal(proposal_id: str):
        p = voting.get_proposal(proposal_id)
        if not p:
            raise HTTPException(404, "Proposal not found")
        return p.model_dump()

    @app.post("/api/pledge")
    async def pledge_resources(req: PledgeRequest, identity=Depends(AuthDependency(sec, required_scope=Scope.WRITE.value))):
        pledge = ResourcePledge(
            user_id=req.user_id,
            node_id=req.node_id,
            cpu_cores=req.cpu_cores,
            gpu_count=req.gpu_count,
            ram_gb=req.ram_gb,
            gpu_vram_mb=req.gpu_vram_mb,
            time_hours=req.time_hours,
            group_id=req.group_id,
        )
        proposal = voting.create_resource_pledge(pledge)
        node_res = NodeResources(
            node_id=req.node_id or req.user_id,
            cpu_cores=req.cpu_cores,
            cpu_available=req.cpu_cores,
            gpu_count=req.gpu_count,
            gpu_available=req.gpu_count,
            ram_gb=req.ram_gb,
            ram_available_gb=req.ram_gb,
            group_id=req.group_id,
        )
        sched.register_node(node_res)
        return {
            "pledge_id": proposal.proposal_id,
            "score": pledge.compute_score,
            "summary": pledge.summary,
        }

    @app.get("/api/pledge/leaderboard")
    async def pledge_leaderboard():
        return {"leaderboard": voting.get_leaderboard()}

    @app.get("/api/resources/cluster")
    async def cluster_resources(group_id: str = ""):
        return voting.get_cluster_resources(group_id)

    @app.post("/api/group/create")
    async def create_group(req: GroupCreateRequest, identity=Depends(AuthDependency(sec, required_scope=Scope.GROUP.value))):
        group = groups.create_group(
            name=req.name,
            owner_id=req.owner_id,
            description=req.description,
            visibility=GroupVisibility(req.visibility),
            policy=GroupPolicy(max_members=req.max_members, require_approval=req.require_approval),
            tags=req.tags,
            passphrase=req.passphrase or None,
        )
        return {"group_id": group.group_id, "name": group.name}

    @app.post("/api/group/join")
    async def join_group(req: GroupJoinRequest, identity=Depends(AuthDependency(sec, required_scope=Scope.GROUP.value))):
        pledge = None
        if req.cpu_cores or req.gpu_count or req.ram_gb:
            pledge = ResourcePledge(
                user_id=req.user_id,
                node_id=req.node_id,
                cpu_cores=req.cpu_cores,
                gpu_count=req.gpu_count,
                ram_gb=req.ram_gb,
                gpu_vram_mb=req.gpu_vram_mb,
                time_hours=req.time_hours,
                group_id=req.group_id,
            )
        ok, msg = groups.join_group(req.group_id, req.user_id, req.node_id, pledge, req.invite_code or None)
        if not ok:
            raise HTTPException(400, msg)
        return {"status": "joined", "group_id": req.group_id, "message": msg}

    @app.post("/api/group/{group_id}/leave")
    async def leave_group(group_id: str, user_id: str = Query(...)):
        ok = groups.leave_group(group_id, user_id)
        if not ok:
            raise HTTPException(400, "Cannot leave group")
        return {"status": "left"}

    @app.get("/api/group/{group_id}")
    async def get_group(group_id: str):
        g = groups.get_group(group_id)
        if not g:
            raise HTTPException(404, "Group not found")
        return groups.get_group_resources(group_id)

    @app.get("/api/group/{group_id}/invite")
    async def create_invite(group_id: str, inviter_id: str = Query(...)):
        code = groups.create_invite(group_id, inviter_id)
        if not code:
            raise HTTPException(403, "Not authorized")
        return {"invite_code": code}

    @app.get("/api/groups")
    async def list_groups(user_id: str = "", visibility: str = ""):
        vis = GroupVisibility(visibility) if visibility else None
        return {"groups": groups.list_groups(user_id or None, vis)}

    @app.post("/api/group/{group_id}/propose-training")
    async def group_propose_training(group_id: str, proposer_id: str = Query(...), model_name: str = "gpt2-small", steps: int = 1000):
        proposal, pid = groups.propose_training(
            group_id=group_id,
            proposer_id=proposer_id,
            job_config={"model_name": model_name, "total_steps": steps},
        )
        if not proposal:
            raise HTTPException(400, pid)
        return {"proposal_id": pid, "status": proposal.status.value}

    @app.post("/api/scheduler/submit")
    async def scheduler_submit(
        name: str = "job",
        min_cpu: int = 1,
        min_gpu: int = 0,
        min_ram: float = 4.0,
        priority: int = JobPriority.NORMAL,
        group_id: str = "",
        submitter: str = "",
        identity=Depends(AuthDependency(sec, required_scope=Scope.TRAIN.value)),
    ):
        req = JobRequirements(
            min_cpu_cores=min_cpu,
            min_gpu_count=min_gpu,
            min_ram_gb=min_ram,
            priority=priority,
            group_id=group_id,
        )
        job_id = sched.submit_job(req, name, submitter)
        assignments = sched.schedule()
        return {"job_id": job_id, "assignments": assignments}

    @app.get("/api/scheduler/status")
    async def scheduler_status():
        return sched.get_queue_status()

    @app.post("/api/github/webhook")
    async def github_webhook(request: Request):
        sig = request.headers.get("X-Hub-Signature-256", "")
        body = await request.body()
        if not gh.verify_webhook(body, sig):
            raise HTTPException(403, "Invalid signature")
        payload = await request.json()
        headers = dict(request.headers)
        event = gh.parse_webhook_event(headers, payload)
        result = await gh.process_webhook(event)
        return result or {"status": "processed"}

    @app.get("/api/demo")
    async def demo():
        return {
            "name": "NetAI",
            "version": "1.0.0",
            "features": [
                "P2P resource discovery",
                "CPU + GPU training",
                "GitHub commit-triggered training",
                "Voting on model selection",
                "Resource pledging",
                "Private group training",
                "Gradient compression & sync",
                "Checkpoint management",
            ],
        }

    @app.post("/api/inference/load")
    async def inference_load_model(req: ModelLoadRequest, identity=Depends(AuthDependency(sec, required_scope=Scope.INFERENCE.value))):
        validator.validate_model_id(req.model_id or req.model_name)
        config = ModelServeConfig(
            model_id=req.model_id or req.model_name,
            model_name=req.model_name,
            version=req.version,
            num_replicas=req.num_replicas,
            num_shards=req.num_shards,
            device=req.device,
            mirror_enabled=req.mirror_enabled,
            max_batch_size=req.max_batch_size,
            group_id=req.group_id,
        )
        try:
            replica = await inf_gw.load_model(config)
            return {
                "model_id": config.model_id,
                "replica_id": replica.replica_id,
                "status": replica.status.value,
                "shards": len(replica.shard_ids),
            }
        except Exception as e:
            return {"error": "Inference gateway error", "request_id": req.model_id}

    @app.post("/api/inference/run")
    async def inference_run(req: InferenceRunRequest, identity=Depends(AuthDependency(sec, required_scope=Scope.INFERENCE.value))):
        validator.validate_prompt(req.prompt)
        validator.validate_positive_int(req.max_tokens, "max_tokens", 1, 8192)
        request = InferenceRequest(
            model_id=req.model_id,
            prompt=req.prompt,
            inputs=req.inputs,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream=req.stream,
            user_id=req.user_id,
            group_id=req.group_id,
            priority=req.priority,
            timeout_ms=req.timeout_ms,
        )
        try:
            response = await inf_gw.serve(request)
            return response.model_dump()
        except Exception as e:
            return {"error": str(e), "request_id": request.request_id}

    @app.get("/api/inference/status")
    async def inference_status():
        return inf_gw.get_status()

    @app.post("/api/inference/unload/{model_id}")
    async def inference_unload(model_id: str, identity=Depends(AuthDependency(sec, required_scope=Scope.INFERENCE.value))):
        result = await inf_engine.unload_model(model_id)
        return {"model_id": model_id, "unloaded": result}

    @app.get("/api/inference/models")
    async def inference_models():
        return {"models": list(inf_engine.models.keys()),
                "details": {mid: {"version": c.version, "shards": c.num_shards}
                           for mid, c in inf_engine.models.items()}}

    @app.get("/api/inference/cache")
    async def inference_cache():
        return inf_lb.mirror.get_status()

    @app.post("/api/inference/node/register")
    async def inference_register_node(node: InferenceNode, identity=Depends(AuthDependency(sec, required_scope=Scope.INFERENCE.value))):
        inf_lb.register_node(node)
        return {"status": "ok", "node_id": node.node_id}

    @app.post("/api/inference/download/{model_id}")
    async def inference_download_model(model_id: str, revision: str = "main"):
        result = await inf_engine.download_and_load_model(model_id, revision=revision)
        if "error" in result:
            raise HTTPException(400, result["error"])
        return result

    @app.post("/api/inference/load-local")
    async def inference_load_local(model_dir: str, model_id: str = "", layer_start: int = -1, layer_end: int = -1):
        if not model_id:
            model_id = os.path.basename(model_dir.rstrip("/"))
        normalized = os.path.normpath(os.path.abspath(model_dir))
        if ".." in model_dir or not os.path.isdir(normalized):
            raise HTTPException(400, f"Invalid directory path")
        result = inf_engine.load_local_model(model_id, normalized, layer_start=layer_start, layer_end=layer_end)
        return result

    @app.post("/api/inference/native-run")
    async def inference_native_run(req: InferenceRunRequest):
        validator.validate_prompt(req.prompt)
        model_id = req.model_id
        engine = inf_engine.get_native_engine()
        model_dir = os.path.join(ENGINE_CACHE_DIR, "models", model_id)
        tok = get_tokenizer(model_dir) if os.path.isdir(model_dir) else None
        if tok and req.prompt:
            prompt_tokens = tok.encode(req.prompt)
        else:
            config = engine.configs.get(model_id, TransformerConfig())
            prompt_tokens = [ord(c) % config.vocab_size for c in req.prompt]
        if not prompt_tokens:
            prompt_tokens = [0]
        result = await inf_engine.native_infer(
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=50,
        )
        if "error" in result:
            raise HTTPException(400, result["error"])
        return result

    @app.get("/api/inference/native/status")
    async def inference_native_status():
        engine = inf_engine.get_native_engine()
        return engine.get_status()

    @app.get("/api/inference/native/models")
    async def inference_native_models():
        engine = inf_engine.get_native_engine()
        downloader = inf_engine.get_model_downloader()
        cached = downloader.list_cached_models()
        return {
            "loaded_models": list(engine._loaded_models),
            "cached_models": cached,
            "status": engine.get_status(),
        }

    @app.delete("/api/inference/native/{model_id}")
    async def inference_native_unload(model_id: str):
        engine = inf_engine.get_native_engine()
        result = engine.unload_model(model_id)
        return {"model_id": model_id, "unloaded": result}

    @app.post("/api/inference/pipeline/plan")
    async def inference_pipeline_plan(model_id: str, node_resources: list[dict] | None = None):
        config = inf_engine.get_native_engine().configs.get(model_id)
        if config is None:
            raise HTTPException(404, f"Model {model_id} not configured — load it first")
        executor = inf_engine.get_pipeline_executor()
        resources = node_resources or [{"node_id": inf_engine.node_id, "vram_available_mb": 8192}]
        stages = executor.plan_pipeline(model_id, config, resources)
        return {
            "model_id": model_id,
            "total_stages": len(stages),
            "stages": [s.model_dump() for s in stages],
        }

    @app.get("/api/inference/pipeline/status")
    async def inference_pipeline_status(model_id: str | None = None):
        executor = inf_engine.get_pipeline_executor()
        if model_id:
            return executor.get_pipeline_status(model_id)
        return {"pipelines": executor.list_pipelines()}

    @app.post("/api/inference/pipeline/activate")
    async def inference_pipeline_activate(req: Request):
        """Remote pipeline stage activation — receives hidden states from
        previous stage, runs assigned layers, returns output activation."""
        from netai.inference.native_engine import _layer_norm
        body = await req.json()
        request_id = body.get("request_id", "")
        model_id = body.get("model_id", "")
        stage_index = body.get("stage_index", 0)
        compressed = body.get("compressed", False)

        engine = inf_engine.get_native_engine()
        if model_id not in engine._loaded_models:
            return JSONResponse({"error": f"Model {model_id} not loaded on this node", "request_id": request_id}, status_code=404)

        shape = body.get("shape", [])
        dtype = body.get("dtype", "float32")
        data_hex = body.get("data_hex", "")

        if compressed:
            from netai.inference.compress import QuantizedTensor
            hidden = ActivationCompressor().decompress(QuantizedTensor(
                data=bytes.fromhex(data_hex), shape=shape,
                dtype_original=dtype, scale=body.get("scale", 1.0),
                zero_point=body.get("zero_point", 0.0), compression_ratio=1.0,
            ))
        else:
            hidden = np.frombuffer(bytes.fromhex(data_hex), dtype=np.dtype(dtype)).reshape(shape)

        t0 = time.time()
        executor = inf_engine.get_pipeline_executor()
        pipeline_stages = executor.pipelines.get(model_id, {})
        my_stage = None
        for sid, s in pipeline_stages.items():
            if s.stage_index == stage_index and s.node_id == engine.node_id:
                my_stage = s
                break

        if my_stage is None:
            return JSONResponse({"error": f"No local stage found for model {model_id}", "request_id": request_id}, status_code=404)

        output = engine.forward(hidden, model_id, my_stage.layer_start, my_stage.layer_end)
        lat_ms = (time.time() - t0) * 1000

        result_data = output.astype(np.float32).tobytes()
        return {
            "request_id": request_id, "model_id": model_id,
            "stage_index": stage_index, "shape": list(output.shape),
            "dtype": str(output.dtype), "data_hex": result_data.hex(),
            "latency_ms": round(lat_ms, 2), "compressed": False,
        }

    @app.get("/api/inference/downloads/status")
    async def inference_download_status():
        downloader = inf_engine.get_model_downloader()
        return {"active_downloads": downloader.get_download_status()}

    @app.post("/api/inference/compress")
    async def inference_compress(req: CompressRequest):
        """Compress activation tensor using 8-bit quantization for pipeline transfer."""
        if not req.data or not req.shape:
            raise HTTPException(400, "data and shape required")
        tensor = np.array(req.data, dtype=np.float32).reshape(req.shape)
        comp = ActivationCompressor(bits=req.bits)
        q = comp.compress(tensor)
        return {
            "data_hex": q.data.hex(),
            "shape": q.shape,
            "dtype": q.dtype_original,
            "scale": float(q.scale),
            "zero_point": float(q.zero_point),
            "compression_ratio": round(float(q.compression_ratio), 2),
            "original_bytes": int(tensor.nbytes),
            "compressed_bytes": int(len(q.data) + 8),
        }

    @app.post("/api/inference/decompress")
    async def inference_decompress(req: Request):
        """Decompress a previously compressed activation tensor."""
        from netai.inference.compress import QuantizedTensor
        body = await req.json()
        data_hex = body.get("data_hex", "")
        shape = body.get("shape", [])
        if not data_hex or not shape:
            raise HTTPException(400, "data_hex and shape required")
        comp = ActivationCompressor(bits=body.get("bits", 8))
        result = comp.decompress(QuantizedTensor(
            data=bytes.fromhex(data_hex), shape=list(shape),
            dtype_original="float32", scale=float(body.get("scale", 1.0)),
            zero_point=float(body.get("zero_point", 0.0)), compression_ratio=1.0,
        ))
        return {"shape": list(result.shape), "dtype": str(result.dtype),
                "data": [float(x) for x in result.flatten().tolist()[:100]]}

    @app.get("/api/inference/native-stream")
    async def inference_native_stream(req: Request):
        """Stream native token-by-token inference via SSE."""
        from fastapi.responses import StreamingResponse
        model_id = req.query_params.get("model_id", "gpt2")
        prompt = req.query_params.get("prompt", "Hello")
        max_tokens = min(int(req.query_params.get("max_tokens", "32")), 128)
        temperature = float(req.query_params.get("temperature", "0.7"))

        async def token_stream():
            from netai.inference.native_engine import _layer_norm
            engine = inf_engine.get_native_engine()
            model_dir = os.path.join(ENGINE_CACHE_DIR, "models", model_id)
            tok = get_tokenizer(model_dir) if os.path.isdir(model_dir) else None
            prompt_tokens = tok.encode(prompt) if tok else [ord(c) % 50257 for c in prompt]
            if not prompt_tokens:
                prompt_tokens = [0]

            config = engine.configs.get(model_id)
            if config is None:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Model {model_id} not loaded'})}\n\n"
                return

            embed = engine.embed_tokens.get(model_id)
            output_w = engine.output_proj.get(model_id)
            ln_f_pair = engine.layer_norm_f.get(model_id)
            tokens = list(prompt_tokens)
            t0 = time.time()

            yield f"data: {json.dumps({'type': 'start', 'model_id': model_id, 'prompt': prompt})}\n\n"
            await asyncio.sleep(0.001)

            for step in range(max_tokens):
                input_ids = np.array(tokens, dtype=np.int64).reshape(1, -1)
                hidden = embed[input_ids]
                for i in range(config.num_layers):
                    hidden = engine.forward_layer(hidden, model_id, i)
                last_hidden = hidden[:, -1, :]
                if ln_f_pair is not None:
                    last_hidden = _layer_norm(last_hidden, ln_f_pair[0], ln_f_pair[1], config.layer_norm_eps)
                logits = last_hidden @ (output_w.T if output_w is not None else embed.T)
                logits = logits / max(temperature, 0.01)
                probs = engine._sample_logits(logits, 50, 0.9)
                next_token = int(np.random.choice(len(probs), p=probs))
                tokens.append(next_token)

                word = tok.decode([next_token]) if tok else str(next_token)
                yield f"data: {json.dumps({'type': 'token', 'token_id': next_token, 'text': word, 'index': step})}\n\n"

            elapsed = (time.time() - t0) * 1000
            all_text = tok.decode(tokens) if tok else " ".join(str(t) for t in tokens)
            gen_count = len(tokens) - len(prompt_tokens)
            yield f"data: {json.dumps({'type': 'done', 'text': all_text, 'tokens_generated': gen_count, 'latency_ms': round(elapsed, 1), 'tokens_per_second': round(gen_count / max(elapsed / 1000, 0.001), 1)})}\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    @app.get("/api/models/catalog")
    async def models_catalog(size_class: str | None = None, min_vram: float | None = None):
        await model_registry.refresh()
        sc = None
        if size_class:
            try:
                sc = ModelSizeClass(size_class)
            except ValueError:
                raise HTTPException(400, f"Invalid size_class. Use: mini, small, mid, large")
        models = model_registry.list_models(sc)
        if min_vram is not None:
            models = model_registry.models_for_vram(min_vram)
        return {
            "version": "1.0.0",
            "total_models": len(model_registry._models),
            "size_class_filter": size_class,
            "min_vram_filter": min_vram,
            "models": [
                {
                    "model_id": m.model_id,
                    "name": m.name,
                    "architecture": m.architecture,
                    "size_class": m.size_class.value,
                    "params_m": m.params_m,
                    "hidden_size": m.hidden_size,
                    "num_layers": m.num_layers,
                    "num_heads": m.num_heads,
                    "vocab_size": m.vocab_size,
                    "intermediate_size": m.intermediate_size,
                    "quantizations": m.quantizations,
                    "vram_required_mb": m.vram_required_mb,
                    "context_length": m.context_length,
                    "license": m.license,
                    "huggingface_id": m.huggingface_id,
                    "description": m.description,
                }
                for m in models
            ],
        }

    @app.get("/api/models/{model_id}")
    async def models_get(model_id: str):
        await model_registry.refresh()
        entry = model_registry.get(model_id)
        if not entry:
            raise HTTPException(404, f"Model '{model_id}' not found in catalog")
        return {
            "model_id": entry.model_id,
            "name": entry.name,
            "architecture": entry.architecture,
            "size_class": entry.size_class.value,
            "params_m": entry.params_m,
            "hidden_size": entry.hidden_size,
            "num_layers": entry.num_layers,
            "num_heads": entry.num_heads,
            "vocab_size": entry.vocab_size,
            "intermediate_size": entry.intermediate_size,
            "quantizations": entry.quantizations,
            "vram_required_mb": entry.vram_required_mb,
            "context_length": entry.context_length,
            "license": entry.license,
            "huggingface_id": entry.huggingface_id,
            "description": entry.description,
            "can_fit_available_vram": entry.can_fit(model_autoloader.available_vram_mb, model_autoloader.preferred_quant),
        }

    @app.get("/api/autoloader/status")
    async def autoloader_status():
        await model_registry.refresh()
        profiler = ResourceProfiler()
        profile = profiler.profile()
        total_vram = sum(profile.gpu_vram_mb) if profile.gpu_vram_mb else 0
        avail_vram = sum(profile.gpu_available_vram_mb) if profile.gpu_available_vram_mb else 0
        model_autoloader.update_resources(avail_vram, available_nodes=max(1, len(await p2p.peer_table.get_alive_peers()) + 1))
        return model_autoloader.get_status()

    @app.post("/api/autoloader/load")
    async def autoloader_load(req: AutoLoaderLoadRequest = None, identity=Depends(AuthDependency(sec, required_scope=Scope.INFERENCE.value))):
        if req is None:
            req = AutoLoaderLoadRequest()
        force_models = req.force_models
        votes = req.votes
        profiler = ResourceProfiler()
        profile = profiler.profile()
        avail_vram = sum(profile.gpu_available_vram_mb) if profile.gpu_available_vram_mb else 0
        model_autoloader.update_resources(avail_vram, available_nodes=max(1, len(await p2p.peer_table.get_alive_peers()) + 1))
        plan = model_autoloader.compute_load_plan(votes=votes, force_models=force_models or None)
        return {"plan": plan, "total_models": len(plan)}

    @app.post("/api/autoloader/loaded/{model_id}")
    async def autoloader_mark_loaded(model_id: str, identity=Depends(AuthDependency(sec, required_scope=Scope.INFERENCE.value))):
        entry = model_registry.get(model_id)
        if not entry:
            raise HTTPException(404, f"Model '{model_id}' not found")
        vram = entry.vram_for_quant(model_autoloader.preferred_quant)
        model_autoloader.mark_loaded(model_id, vram)
        config = ModelServeConfig(
            model_id=entry.model_id,
            model_name=entry.name,
            num_shards=1,
            quantization=model_autoloader.preferred_quant,
        )
        try:
            replica = await inf_gw.load_model(config)
            return {"model_id": model_id, "vram_mb": vram, "status": "loaded", "replica_id": replica.replica_id}
        except Exception:
            return {"model_id": model_id, "vram_mb": vram, "status": "loaded_in_catalog_only"}

    @app.delete("/api/autoloader/loaded/{model_id}")
    async def autoloader_mark_unloaded(model_id: str, identity=Depends(AuthDependency(sec, required_scope=Scope.INFERENCE.value))):
        model_autoloader.mark_unloaded(model_id)
        if model_id in inf_engine.models:
            await inf_engine.unload_model(model_id)
        return {"model_id": model_id, "status": "unloaded"}

    @app.get("/api/autoloader/recommend")
    async def autoloader_recommend(vram_mb: float | None = None):
        await model_registry.refresh()
        if vram_mb is not None:
            profiler = ResourceProfiler()
            profile = profiler.profile()
            avail = vram_mb
        else:
            profiler = ResourceProfiler()
            profile = profiler.profile()
            avail = sum(profile.gpu_available_vram_mb) if profile.gpu_available_vram_mb else 0
        return {
            "available_vram_mb": avail,
            "recommended_models": [
                {
                    "model_id": m.model_id,
                    "name": m.name,
                    "size_class": m.size_class.value,
                    "params_m": m.params_m,
                    "vram_mb": m.vram_for_quant("q4_k_m"),
                    "quant": "q4_k_m",
                }
                for m in model_registry.models_for_vram(avail, "q4_k_m")
            ],
        }

    @app.post("/api/jack-in")
    async def jack_in(req: JackInRequest, identity=Depends(AuthDependency(sec, required_scope=Scope.WRITE.value))):
        profiler = ResourceProfiler()
        profile = profiler.profile()
        results = {"user_id": req.user_id, "node_id": req.node_id or p2p.node_id, "modes": []}
        if req.mode in ("training", "both"):
            pledge = ResourcePledge(
                user_id=req.user_id,
                node_id=req.node_id or p2p.node_id,
                cpu_cores=req.cpu_cores or profile.cpu_available,
                gpu_count=req.gpu_count or profile.gpu_available,
                ram_gb=req.ram_gb or profile.ram_available_gb,
                gpu_vram_mb=req.gpu_vram_mb or profile.gpu_available_vram_mb,
                time_hours=req.time_hours,
                group_id=req.group_id,
            )
            proposal = voting.create_resource_pledge(pledge)
            node_res = NodeResources(
                node_id=req.node_id or p2p.node_id,
                cpu_cores=pledge.cpu_cores,
                cpu_available=pledge.cpu_cores,
                gpu_count=pledge.gpu_count,
                gpu_available=pledge.gpu_count,
                ram_gb=pledge.ram_gb,
                ram_available_gb=pledge.ram_gb,
                group_id=req.group_id,
            )
            sched.register_node(node_res)
            results["modes"].append("training")
            results["pledge_score"] = pledge.compute_score
            results["pledge_summary"] = pledge.summary
            results["training_pledge_id"] = proposal.proposal_id
            if req.group_id:
                groups.join_group(req.group_id, req.user_id, req.node_id or p2p.node_id, pledge)
                results["group_joined"] = req.group_id
        if req.mode in ("inference", "both"):
            inf_node = InferenceNode(
                node_id=req.node_id or p2p.node_id,
                endpoint=f"http://{p2p._get_local_ip()}:{p2p.port}",
                status=InferenceStatus.READY,
                models_loaded=req.models_to_serve,
                gpu_count=req.gpu_count or profile.gpu_available,
                gpu_available=req.gpu_count or profile.gpu_available,
                cpu_cores=req.cpu_cores or profile.cpu_available,
                ram_gb=req.ram_gb or profile.ram_available_gb,
                group_id=req.group_id,
            )
            inf_lb.register_node(inf_node)
            results["modes"].append("inference")
            results["inference_node_id"] = inf_node.node_id
            for model_id in req.models_to_serve:
                config = ModelServeConfig(model_id=model_id, model_name=model_id)
                try:
                    replica = await inf_gw.load_model(config)
                    results[f"model_{model_id}_loaded"] = True
                except Exception:
                    results[f"model_{model_id}_loaded"] = False
        results["profile"] = {
            "cpu": f"{profile.cpu_available}/{profile.cpu_cores}",
            "gpu": f"{profile.gpu_available}/{profile.gpu_count}",
            "ram": f"{profile.ram_available_gb:.1f}/{profile.ram_total_gb:.1f}GB",
        }
        return results

    @app.post("/api/auth/register")
    async def register_user(req: RegisterRequest, identity=Depends(AuthDependency(sec, allow_unauthenticated=True))):
        try:
            role = UserRole(req.role)
            if identity is None and role.value not in ("user",):
                raise HTTPException(403, "Unauthenticated registration requires role=user")
            user = sec.register_user(req.user_id, req.password, role, scopes=req.scopes or None)
            return {"user_id": user.user_id, "role": user.role.value, "scopes": user.scopes}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/auth/login")
    async def login(req: LoginRequest):
        user = sec.authenticate_password(req.user_id, req.password)
        if not user:
            sec.audit.log("auth_failure", user_id=req.user_id, risk_score=0.4)
            raise HTTPException(401, "Invalid credentials")
        token = sec.create_token(req.user_id, user.scopes, ttl_hours=24.0)
        sec.audit.log("auth_login", user_id=req.user_id, status="ok")
        return {"access_token": token.token, "refresh_token": token.refresh_token,
                "expires_in_hours": 24.0, "scopes": token.scopes, "user_id": req.user_id}

    @app.post("/api/auth/token")
    async def create_auth_token(req: AuthTokenRequest, identity=Depends(AuthDependency(sec, required_scope=Scope.WRITE.value))):
        try:
            token = sec.create_token(req.user_id, req.scopes, req.ttl_hours)
            return {"token": token.token, "refresh_token": token.refresh_token,
                    "expires_in_hours": req.ttl_hours, "scopes": token.scopes}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/auth/verify")
    async def verify_auth_token(token: str = Query(...)):
        record = sec.verify_token(token)
        if not record:
            raise HTTPException(401, "Invalid or expired token")
        return {"valid": True, "user_id": record.user_id, "scopes": record.scopes}

    @app.post("/api/auth/api-key")
    async def create_api_key(req: ApiKeyRequest, identity=Depends(AuthDependency(sec, required_scope=Scope.ADMIN.value))):
        try:
            key = sec.create_api_key(req.user_id, req.name, req.scopes)
            return {"api_key": key.key, "name": key.name, "scopes": key.scopes}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/auth/revoke-token")
    async def revoke_auth_token(token: str = Query(...), identity=Depends(AuthDependency(sec, required_scope=Scope.ADMIN.value))):
        ok = sec.revoke_token(token)
        return {"revoked": ok}

    @app.post("/api/auth/revoke-key")
    async def revoke_api_key(key: str = Query(...), identity=Depends(AuthDependency(sec, required_scope=Scope.ADMIN.value))):
        ok = sec.revoke_api_key(key)
        return {"revoked": ok}

    @app.get("/api/auth/users")
    async def list_users(identity=Depends(AuthDependency(sec, required_scope=Scope.ADMIN.value))):
        return {"users": [
            {"user_id": u.user_id, "role": u.role.value, "scopes": u.scopes,
             "disabled": u.disabled, "last_login": u.last_login}
            for u in sec.users.values()
        ]}

    @app.get("/api/security/status")
    async def security_status(identity=Depends(AuthDependency(sec, required_scope=Scope.ADMIN.value))):
        return {
            "security": sec.get_security_status(),
            "gradient_integrity": grad_integrity.get_status(),
            "model_provenance": model_prov.get_status(),
        }

    @app.get("/api/security/audit")
    async def audit_log(limit: int = Query(100), event_type: str = Query(""),
                        identity=Depends(AuthDependency(sec, required_scope=Scope.ADMIN.value))):
        events = sec.audit.get_recent(limit=limit, event_type=event_type or None)
        return {"events": events}

    @app.get("/api/security/alerts")
    async def security_alerts(identity=Depends(AuthDependency(sec, required_scope=Scope.ADMIN.value))):
        return {"alerts": sec.audit.get_alerts()}

    @app.get("/api/metrics")
    async def prometheus_metrics():
        metrics_data["requests"] += 1
        lines = []
        lines.append("# HELP nx_requests_total Total API requests")
        lines.append("# TYPE nx_requests_total counter")
        lines.append(f"nx_requests_total {metrics_data['requests']}")
        lines.append("# HELP nx_errors_total Total API errors")
        lines.append("# TYPE nx_errors_total counter")
        lines.append(f"nx_errors_total {metrics_data['errors']}")
        lines.append("# HELP nx_uptime_seconds Server uptime in seconds")
        lines.append("# TYPE nx_uptime_seconds gauge")
        lines.append(f"nx_uptime_seconds {time.time() - metrics_data['start_time']:.0f}")
        prof = ResourceProfiler()
        profile = prof.profile()
        lines.append("# HELP nx_cpu_cores_total Total CPU cores")
        lines.append("# TYPE nx_cpu_cores_total gauge")
        lines.append(f"nx_cpu_cores_total {profile.cpu_cores}")
        lines.append("# HELP nx_cpu_available Available CPU cores")
        lines.append("# TYPE nx_cpu_available gauge")
        lines.append(f"nx_cpu_available {profile.cpu_available}")
        lines.append("# HELP nx_gpu_count GPU count")
        lines.append("# TYPE nx_gpu_count gauge")
        lines.append(f"nx_gpu_count {profile.gpu_count}")
        lines.append("# HELP nx_ram_total_gb Total RAM in GB")
        lines.append("# TYPE nx_ram_total_gb gauge")
        lines.append(f"nx_ram_total_gb {profile.ram_total_gb:.1f}")
        lines.append("# HELP nx_ram_available_gb Available RAM in GB")
        lines.append("# TYPE nx_ram_available_gb gauge")
        lines.append(f"nx_ram_available_gb {profile.ram_available_gb:.1f}")
        lines.append("# HELP nx_vram_total_mb Total GPU VRAM in MB")
        lines.append("# TYPE nx_vram_total_mb gauge")
        vram = sum(profile.gpu_vram_mb) if profile.gpu_vram_mb else 0
        lines.append(f"nx_vram_total_mb {vram}")
        lines.append("# HELP nx_peers_connected Connected peers")
        lines.append("# TYPE nx_peers_connected gauge")
        peers = await p2p.peer_table.get_alive_peers()
        lines.append(f"nx_peers_connected {len(peers)}")
        lines.append("# HELP nx_training_jobs_active Active training jobs")
        lines.append("# TYPE nx_training_jobs_active gauge")
        jobs = coord.list_jobs()
        active = sum(1 for j in jobs if j.get("status") in ("running", "queued"))
        lines.append(f"nx_training_jobs_active {active}")
        lines.append("# HELP nx_inferences_total Total inferences")
        lines.append("# TYPE nx_inferences_total counter")
        lines.append(f"nx_inferences_total {len(inf_engine.metrics_history)}")
        lines.append("# HELP nx_inference_models_loaded Loaded inference models")
        lines.append("# TYPE nx_inference_models_loaded gauge")
        lines.append(f"nx_inference_models_loaded {len(inf_engine.models)}")
        lines.append("# HELP nx_groups_total Total groups")
        lines.append("# TYPE nx_groups_total gauge")
        lines.append(f"nx_groups_total {len(groups.groups)}")
        lines.append("# HELP nx_proposals_total Total proposals")
        lines.append("# TYPE nx_proposals_total gauge")
        lines.append(f"nx_proposals_total {len(voting.proposals)}")
        lines.append("# HELP nx_pledges_total Total pledges")
        lines.append("# TYPE nx_pledges_total gauge")
        lines.append(f"nx_pledges_total {len(voting.pledges)}")
        lines.append("# HELP nx_inference_nodes_total Total inference nodes")
        lines.append("# TYPE nx_inference_nodes_total gauge")
        lines.append(f"nx_inference_nodes_total {len(inf_lb.nodes)}")
        lines.append("# HELP nx_inference_nodes_available Available inference nodes")
        lines.append("# TYPE nx_inference_nodes_available gauge")
        avail = sum(1 for n in inf_lb.nodes.values() if n.is_available)
        lines.append(f"nx_inference_nodes_available {avail}")
        inf_status = inf_gw.get_status()
        local_inf = inf_status.get("local", {})
        lines.append("# HELP nx_inference_local_status Local inference status")
        lines.append('# TYPE nx_inference_local_status gauge')
        lines.append(f'nx_inference_local_status{{status="{local_inf.get("status","unknown")}"}} 1')
        return Response(content="\n".join(lines) + "\n", media_type="text/plain")

    @app.websocket("/ws/inference/stream")
    async def websocket_inference_stream(websocket: WebSocket):
        token = websocket.query_params.get("token") or websocket.headers.get("Authorization", "").replace("Bearer ", "")
        if not token or not sec.verify_token(token):
            await websocket.close(code=4001, reason="Unauthorized")
            return
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                model_id = data.get("model_id", "")
                prompt = data.get("prompt", "")
                max_tokens = data.get("max_tokens", 256)
                temperature = data.get("temperature", 0.7)
                top_p = data.get("top_p", 0.9)
                user_id = data.get("user_id", "")
                group_id = data.get("group_id", "")
                request = InferenceRequest(
                    model_id=model_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    user_id=user_id,
                    group_id=group_id,
                )
                async for chunk in inf_gw.stream_serve(request):
                    await websocket.send_json(chunk)
                    if chunk.get("type") in ("done", "error"):
                        break
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning("WebSocket inference error: %s", e)

    @app.websocket("/ws/inference/stream-native")
    async def websocket_native_stream(websocket: WebSocket):
        """WebSocket token-by-token streaming via NativeInferenceEngine.

        Accepts JSON: {model_id, prompt, max_tokens, temperature, top_p}
        Streams: {type: "token"/"done"/"error", text, token_id, ...}
        Supports {"type": "cancel"} to abort generation.
        """
        await websocket.accept()
        from netai.inference.native_engine import _layer_norm
        engine = inf_engine.get_native_engine()
        cancel_event = asyncio.Event()

        try:
            data = await websocket.receive_json()
            model_id = str(data.get("model_id", ""))
            prompt = str(data.get("prompt", ""))
            max_tokens = min(int(data.get("max_tokens", 64)), 4096)
            temperature = float(data.get("temperature", 0.7))
            top_p = float(data.get("top_p", 0.9))
        except (WebSocketDisconnect, ValueError, TypeError) as e:
            try:
                await websocket.send_json({"type": "error", "error": f"Invalid request: {e}"})
            except Exception:
                pass
            return

        config = engine.configs.get(model_id)
        if config is None:
            await websocket.send_json({"type": "error", "error": f"Model {model_id} not loaded"})
            return

        model_dir = os.path.join(ENGINE_CACHE_DIR, "models", model_id)
        tok = get_tokenizer(model_dir) if os.path.isdir(model_dir) else None
        prompt_tokens = tok.encode(prompt) if tok else [ord(c) % config.vocab_size for c in prompt]
        if not prompt_tokens:
            prompt_tokens = [0]

        embed = engine.embed_tokens.get(model_id)
        output_w = engine.output_proj.get(model_id)
        ln_f_pair = engine.layer_norm_f.get(model_id)
        tokens = list(prompt_tokens)
        t0 = time.time()

        await websocket.send_json({
            "type": "start",
            "model_id": model_id,
            "prompt": prompt,
            "prompt_tokens": len(prompt_tokens),
        })

        cancelled = False
        try:
            for step in range(max_tokens):
                try:
                    pending = await asyncio.wait_for(
                        websocket.receive_text(), timeout=0.001
                    )
                    try:
                        msg = json.loads(pending)
                        if msg.get("type") == "cancel":
                            cancelled = True
                            await websocket.send_json({
                                "type": "cancelled",
                                "tokens_generated": step,
                                "text": tok.decode(tokens) if tok else " ".join(str(t) for t in tokens),
                            })
                            break
                    except (json.JSONDecodeError, TypeError):
                        pass
                except (asyncio.TimeoutError, WebSocketDisconnect):
                    pass

                if cancel_event.is_set():
                    cancelled = True
                    break

                input_ids = np.array(tokens, dtype=np.int64).reshape(1, -1)
                hidden = embed[input_ids]
                for i in range(config.num_layers):
                    hidden = engine.forward_layer(hidden, model_id, i)
                last_hidden = hidden[:, -1, :]
                if ln_f_pair is not None:
                    last_hidden = _layer_norm(last_hidden, ln_f_pair[0], ln_f_pair[1], config.layer_norm_eps)
                logits = last_hidden @ (output_w.T if output_w is not None else embed.T)
                logits = logits / max(temperature, 0.01)
                probs = engine._sample_logits(logits, 50, top_p)
                next_token = int(np.random.choice(len(probs), p=probs))
                tokens.append(next_token)

                word = tok.decode([next_token]) if tok else str(next_token)
                await websocket.send_json({
                    "type": "token",
                    "token_id": next_token,
                    "text": word,
                    "index": step,
                })

            elapsed = (time.time() - t0) * 1000
            all_text = tok.decode(tokens) if tok else " ".join(str(t) for t in tokens)
            gen_count = len(tokens) - len(prompt_tokens)

            if not cancelled:
                await websocket.send_json({
                    "type": "done",
                    "text": all_text,
                    "tokens_generated": gen_count,
                    "latency_ms": round(elapsed, 1),
                    "tokens_per_second": round(gen_count / max(elapsed / 1000, 0.001), 1),
                })

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected during native stream")
        except Exception as e:
            logger.error("Native WebSocket stream error: %s", e)
            try:
                await websocket.send_json({"type": "error", "error": str(e)})
            except Exception:
                pass

    @app.get("/api/inference/stream")
    async def sse_inference_stream(
        model_id: str = Query(...),
        prompt: str = Query(...),
        max_tokens: int = Query(256),
        temperature: float = Query(0.7),
        top_p: float = Query(0.9),
        identity=Depends(AuthDependency(sec, required_scope=Scope.INFERENCE.value)),
    ):
        if max_tokens < 1 or max_tokens > 4096:
            raise HTTPException(400, "max_tokens must be 1-4096")
        if not 0 <= temperature <= 2.0:
            raise HTTPException(400, "temperature must be 0-2.0")
        if not 0 <= top_p <= 1.0:
            raise HTTPException(400, "top_p must be 0-1.0")
        if len(prompt) > 32768:
            raise HTTPException(400, "Prompt too long (max 32K chars)")

        async def event_generator():
            request = InferenceRequest(
                model_id=model_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            async for chunk in inf_gw.stream_serve(request):
                yield f"data: {json.dumps(chunk)}\n\n"
                if chunk.get("type") in ("done", "error"):
                    break
            yield "data: {\"type\": \"stream_end\"}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.get("/api/inference/stream-sse")
    async def sse_inference_stream_post(request: Request):
        try:
            max_tokens = int(request.query_params.get("max_tokens", "256"))
            temperature = float(request.query_params.get("temperature", "0.7"))
            top_p = float(request.query_params.get("top_p", "0.9"))
        except (ValueError, TypeError):
            raise HTTPException(400, "Invalid numeric parameter")
        return await sse_inference_stream(
            model_id=request.query_params.get("model_id", ""),
            prompt=request.query_params.get("prompt", ""),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    @app.post("/api/training/gradient-sync")
    async def gradient_sync_endpoint(payload: GradientSyncPayload, identity=Depends(AuthDependency(sec, required_scope=Scope.GRADIENT.value))):
        if payload.gradient_hash and payload.gradients:
            try:
                validator.validate_gradient_data(payload.gradients)
            except ValueError as e:
                sec.audit.log("gradient_validation_fail", details={"error": str(e)}, risk_score=0.5)
                raise HTTPException(400, str(e))
        ok = await grad_sync.receive_gradients(payload.model_dump())
        if ok:
            return {"status": "ok", "job_id": payload.job_id, "step": payload.step}
        return {"status": "error", "message": "No gradient data received"}

    @app.post("/api/training/gradient-push/{job_id}/{step}")
    async def gradient_push(job_id: str, step: int, identity=Depends(AuthDependency(sec, required_scope=Scope.GRADIENT.value))):
        stored = await grad_sync.pull_aggregated(job_id, step)
        if stored is None:
            job = coord.jobs.get(job_id)
            if job:
                gradients = await job.aggregate_gradients(step)
                if gradients:
                    await grad_sync.push_gradients(job_id, step, gradients)
                    return {"status": "pushed", "job_id": job_id, "step": step, "layers": len(gradients)}
            return {"status": "no_gradients", "job_id": job_id, "step": step}
        await grad_sync.push_gradients(job_id, step, stored)
        return {"status": "pushed", "job_id": job_id, "step": step, "layers": len(stored)}

    @app.get("/api/training/gradient-pull/{job_id}/{step}")
    async def gradient_pull(job_id: str, step: int, identity=Depends(AuthDependency(sec, required_scope=Scope.READ.value))):
        result = await grad_sync.pull_aggregated(job_id, step)
        if result is None:
            raise HTTPException(404, "No aggregated gradients for this step")
        serialized = {layer: {"shape": list(arr.shape), "mean": float(arr.mean()), "std": float(arr.std()), "norm": float(np.linalg.norm(arr))} for layer, arr in result.items()}
        return {"job_id": job_id, "step": step, "gradients": serialized}

    @app.post("/api/training/gradient-aggregate/{job_id}/{step}")
    async def gradient_aggregate(job_id: str, step: int, identity=Depends(AuthDependency(sec, required_scope=Scope.GRADIENT.value))):
        result = await grad_sync.aggregate_for_step(job_id, step)
        if not result:
            return {"status": "no_data", "job_id": job_id, "step": step}
        return {"status": "aggregated", "job_id": job_id, "step": step, "layers": len(result)}

    @app.get("/api/training/gradient-status")
    async def gradient_status(identity=Depends(AuthDependency(sec, required_scope=Scope.READ.value))):
        return grad_sync.get_sync_status()

    @app.post("/api/training/gradient-peer")
    async def add_gradient_peer(node_id: str = Query(...), endpoint: str = Query(...), identity=Depends(AuthDependency(sec, required_scope=Scope.GRADIENT.value))):
        grad_sync.add_peer(node_id, endpoint)
        return {"status": "ok", "node_id": node_id, "endpoint": endpoint}

    # ── Cache Management Endpoints ──────────────────────────────────────────
    cache_manager = ModelCacheManager(downloader=inf_engine.get_model_downloader())

    @app.get("/api/cache/models")
    async def cache_list_models():
        return {"models": cache_manager.list_models()}

    @app.get("/api/cache/models/{model_id}")
    async def cache_model_info(model_id: str):
        info = cache_manager.get_model_info(model_id)
        if info is None:
            raise HTTPException(404, f"Model '{model_id}' not found in cache")
        return info

    @app.delete("/api/cache/models/{model_id}")
    async def cache_delete_model(model_id: str, identity=Depends(AuthDependency(sec, required_scope=Scope.WRITE.value))):
        result = cache_manager.delete_model(model_id)
        if result.get("status") == "not_found":
            raise HTTPException(404, f"Model '{model_id}' not found in cache")
        return result

    @app.post("/api/cache/prune")
    async def cache_prune(keep_latest_n: int = Query(5, ge=1, le=100), identity=Depends(AuthDependency(sec, required_scope=Scope.WRITE.value))):
        result = cache_manager.prune_cache(keep_latest_n)
        return result

    @app.get("/api/cache/stats", response_model=CacheStatsResponse)
    async def cache_stats():
        return cache_manager.cache_stats()

    @app.post("/api/cache/verify/{model_id}")
    async def cache_verify(model_id: str):
        result = cache_manager.verify_integrity(model_id)
        if result.get("status") == "not_found":
            raise HTTPException(404, f"Model '{model_id}' not found in cache")
        return result

    @app.get("/api/cache/search")
    async def cache_search(q: str = Query(..., min_length=1)):
        results = cache_manager.search_models(q)
        return {"query": q, "count": len(results), "models": results}

    @app.get("/api/cache/export")
    async def cache_export():
        return cache_manager.export_model_info()

    @app.post("/api/cache/hit")
    async def cache_hit(req: CacheHitRequest):
        result = cache_manager.cache_hit(req.model_id)
        return result

    @app.post("/api/benchmark/run/{model_id}")
    async def benchmark_run(model_id: str, config: BenchmarkConfig | None = None):
        """Run full benchmark suite for a model."""
        if config is None:
            config = BenchmarkConfig(model_id=model_id)
        else:
            config.model_id = model_id
        result = benchmark.run_suite(model_id, config)
        if result.error:
            return {"status": "error", "model_id": model_id, "error": result.error}
        return {"status": "ok", "model_id": model_id, "result": result.model_dump()}

    @app.get("/api/benchmark/report/{model_id}")
    async def benchmark_report(model_id: str):
        """Get Markdown benchmark report for a model."""
        if model_id not in benchmark.results:
            result = benchmark.run_suite(model_id)
            if result.error:
                raise HTTPException(500, f"Benchmark failed: {result.error}")
        report = benchmark.generate_report(model_id)
        return {"model_id": model_id, "report": report}

    @app.get("/api/benchmark/compare")
    async def benchmark_compare(models: str = Query(default="")):
        """Compare models side-by-side. Pass comma-separated model IDs."""
        if not models:
            raise HTTPException(400, "models query parameter required (comma-separated)")
        model_ids = [m.strip() for m in models.split(",") if m.strip()]
        if not model_ids:
            raise HTTPException(400, "No valid model IDs")
        comparison = benchmark.compare_models(model_ids)
        return {"comparison": comparison}

    @app.get("/api/benchmark/stats")
    async def benchmark_stats():
        """Get cache statistics: total size, model count, hit rate, disk free."""
        cache_models_dir = os.path.join(ENGINE_CACHE_DIR, "models")
        total_size = 0.0
        model_count = 0
        if os.path.isdir(cache_models_dir):
            for entry in os.listdir(cache_models_dir):
                entry_path = os.path.join(cache_models_dir, entry)
                if os.path.isdir(entry_path):
                    model_count += 1
                    for dirpath, _dirs, filenames in os.walk(entry_path):
                        for fname in filenames:
                            fpath = os.path.join(dirpath, fname)
                            try:
                                total_size += os.path.getsize(fpath)
                            except OSError:
                                pass

        disk_free = 0.0
        try:
            statvfs = os.statvfs(ENGINE_CACHE_DIR)
            disk_free = (statvfs.f_frsize * statvfs.f_bavail) / (1024 ** 3)
        except (OSError, AttributeError):
            pass

        return {
            "cache_dir": ENGINE_CACHE_DIR,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "model_count": model_count,
            "disk_free_gb": round(disk_free, 2),
        }

    @app.get("/api/ws/test", response_class=HTMLResponse)
    async def websocket_test_page():
        return _ws_test_html()

    return app


def _dashboard_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NetAI - Distributed AI Inference</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:#0a0a0f;color:#e0e0e0;min-height:100vh}
.header{background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%);padding:20px 32px;border-bottom:1px solid #21262d;display:flex;align-items:center;gap:16px}
.header h1{font-size:24px;color:#58a6ff;font-weight:700;letter-spacing:-0.5px}
.header h1 span{color:#8b949e;font-weight:400;font-size:14px;margin-left:8px}
.header .node-id{color:#484f58;font-size:12px;font-family:monospace;margin-left:auto}
.tabs{display:flex;background:#0d1117;border-bottom:1px solid #21262d;padding:0 24px;gap:0;overflow-x:auto}
.tab{padding:12px 20px;cursor:pointer;color:#8b949e;font-size:13px;font-weight:500;border-bottom:2px solid transparent;transition:all .15s;white-space:nowrap}
.tab:hover{color:#c9d1d9;background:#161b22}
.tab.active{color:#58a6ff;border-bottom-color:#58a6ff;background:#161b22}
.content{padding:24px 32px;max-width:1400px;margin:0 auto}
.panel{display:none}.panel.active{display:block}
.card{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:20px;margin-bottom:16px}
.card h3{color:#58a6ff;font-size:15px;margin-bottom:12px;font-weight:600}
.stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin-bottom:20px}
.stat{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px;text-align:center}
.stat .value{font-size:28px;font-weight:700;color:#58a6ff}
.stat .label{font-size:11px;color:#8b949e;margin-top:2px;text-transform:uppercase;letter-spacing:.5px}
table{width:100%;border-collapse:collapse}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #21262d;font-size:13px}
th{color:#8b949e;font-weight:600;text-transform:uppercase;font-size:10px;letter-spacing:.5px}
tr:hover{background:#1c2128}
.btn{padding:8px 16px;border:1px solid #30363d;background:#21262d;color:#c9d1d9;border-radius:6px;cursor:pointer;font-size:13px;font-weight:500;transition:all .15s}
.btn:hover{background:#30363d;border-color:#58a6ff;color:#fff}
.btn-primary{background:#1f6feb;border-color:#1f6feb;color:#fff}
.btn-primary:hover{background:#388bfd}
.btn-danger{background:#da3633;border-color:#da3633;color:#fff}
.btn-danger:hover{background:#f85149}
.btn-sm{padding:4px 10px;font-size:12px}
.form-group{margin-bottom:12px}
.form-group label{display:block;font-size:11px;color:#8b949e;margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px;font-weight:600}
.form-group input,.form-group select,.form-group textarea{width:100%;padding:8px 12px;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e0e0e0;font-size:14px;font-family:inherit}
.form-group input:focus,.form-group select:focus,.form-group textarea:focus{outline:none;border-color:#58a6ff;box-shadow:0 0 0 1px #1f6feb33}
.form-row{display:flex;gap:12px;flex-wrap:wrap}
.form-row .form-group{flex:1;min-width:140px}
.output-box{margin-top:12px;padding:16px;background:#0d1117;border:1px solid #21262d;border-radius:6px;font-family:'JetBrains Mono','Fira Code',monospace;font-size:13px;white-space:pre-wrap;word-break:break-all;max-height:400px;overflow-y:auto;color:#7ee787;display:none}
.output-box.error{color:#f85149}
.output-box.visible{display:block}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600}
.badge-success{background:#1b4332;color:#7ee787}
.badge-warning{background:#4a3000;color:#d29922}
.badge-error{background:#4c1111;color:#f85149}
.badge-info{background:#0c2d6b;color:#58a6ff}
.badge-neutral{background:#21262d;color:#8b949e}
.model-card{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:16px;margin-bottom:12px}
.model-card .model-name{color:#58a6ff;font-size:16px;font-weight:600}
.model-card .model-meta{color:#8b949e;font-size:12px;margin-top:4px}
.gen-output{background:#0d1117;border:1px solid #238636;border-radius:6px;padding:16px;margin-top:12px;font-family:'JetBrains Mono',monospace;font-size:14px;line-height:1.6;color:#e6edf3;white-space:pre-wrap;max-height:300px;overflow-y:auto}
.gen-meta{display:flex;gap:16px;margin-top:8px;color:#8b949e;font-size:12px}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid #30363d;border-top-color:#58a6ff;border-radius:50%;animation:spin .8s linear infinite;margin-right:8px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="header">
  <h1>⚡ NetAI <span>Distributed AI Inference</span></h1>
  <div class="node-id" id="node-id">connecting...</div>
</div>
<div class="tabs">
  <div class="tab active" onclick="showTab('overview')">Overview</div>
  <div class="tab" onclick="showTab('models')">Models</div>
  <div class="tab" onclick="showTab('generate')">Generate</div>
  <div class="tab" onclick="showTab('download')">Download</div>
  <div class="tab" onclick="showTab('pipeline')">Pipeline</div>
  <div class="tab" onclick="showTab('network')">Network</div>
</div>
<div class="content">
  <div id="toast-container" style="position:fixed;top:16px;right:16px;z-index:9999"></div>

  <!-- OVERVIEW -->
  <div id="overview" class="panel active">
    <div class="stats" id="ov-stats"></div>
    <div class="card">
      <h3>🖥️ Hardware Profile</h3>
      <div id="ov-hardware">Loading...</div>
    </div>
    <div class="card">
      <h3>🧠 Native Engine</h3>
      <div id="ov-native">Loading...</div>
    </div>
  </div>

  <!-- MODELS -->
  <div id="models" class="panel">
    <div class="card">
      <h3>📥 Loaded Models</h3>
      <div id="models-loaded">Loading...</div>
    </div>
    <div class="card">
      <h3>📂 Cached Models</h3>
      <div id="models-cached">Loading...</div>
    </div>
    <div class="card">
      <h3>📋 Model Catalog</h3>
      <div id="models-catalog">Loading...</div>
    </div>
  </div>

  <!-- GENERATE -->
  <div id="generate" class="panel">
    <div class="card">
      <h3>🧠 Native Inference</h3>
      <p style="color:#8b949e;margin-bottom:16px;font-size:13px">Run real transformer inference on models loaded into the native engine. No external runtime required.</p>
      <div class="form-group">
        <label>Model ID</label>
        <select id="gen-model"><option value="">-- Select a loaded model --</option></select>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Prompt</label><textarea id="gen-prompt" rows="4" placeholder="Enter your prompt...">The meaning of life is</textarea></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Max Tokens</label><input id="gen-max-tokens" type="number" value="64" min="1" max="4096"></div>
        <div class="form-group"><label>Temperature</label><input id="gen-temp" type="number" value="0.7" min="0" max="2" step="0.1"></div>
        <div class="form-group"><label>Top-P</label><input id="gen-topp" type="number" value="0.9" min="0" max="1" step="0.05"></div>
      </div>
      <button class="btn btn-primary" onclick="generateNative()" id="gen-btn">⚡ Generate</button>
      <button class="btn" onclick="loadLocalModel()" style="margin-left:8px">📂 Load Model from Disk</button>
      <div id="gen-output" class="output-box"></div>
      <div id="gen-meta" class="gen-meta" style="display:none"></div>
    </div>
  </div>

  <!-- DOWNLOAD -->
  <div id="download" class="panel">
    <div class="card">
      <h3>🌐 Download from HuggingFace</h3>
      <p style="color:#8b949e;margin-bottom:16px;font-size:13px">Download open-source models directly from HuggingFace. Only MIT/Apache-compatible licenses are allowed.</p>
      <div class="form-row">
        <div class="form-group"><label>Model ID (e.g. gpt2, facebook/opt-125m)</label><input id="dl-model" value="gpt2"></div>
        <div class="form-group"><label>Revision</label><input id="dl-revision" value="main"></div>
      </div>
      <button class="btn btn-primary" onclick="downloadModel()" id="dl-btn">📥 Download & Load</button>
      <div id="dl-output" class="output-box"></div>
    </div>
    <div class="card">
      <h3>📂 Load from Local Directory</h3>
      <p style="color:#8b949e;margin-bottom:16px;font-size:13px">Load a model from a local directory containing config.json and model weights.</p>
      <div class="form-row">
        <div class="form-group"><label>Directory Path</label><input id="ll-dir" value="/path/to/model"></div>
        <div class="form-group"><label>Model ID</label><input id="ll-modelid" value="my-model"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Layer Start (-1 = all)</label><input id="ll-layer-start" type="number" value="-1"></div>
        <div class="form-group"><label>Layer End (-1 = all)</label><input id="ll-layer-end" type="number" value="-1"></div>
      </div>
      <button class="btn btn-primary" onclick="loadLocalModel()">📂 Load Model</button>
    </div>
  </div>

  <!-- PIPELINE -->
  <div id="pipeline" class="panel">
    <div class="card">
      <h3>🔀 Pipeline-Parallel Plan</h3>
      <p style="color:#8b949e;margin-bottom:16px;font-size:13px">Distribute transformer layers across multiple nodes. Each node runs a contiguous range of layers.</p>
      <div class="form-row">
        <div class="form-group"><label>Model ID</label><input id="pipe-model" value="gpt2"></div>
        <div class="form-group"><label>VRAM per node (MB)</label><input id="pipe-vram" type="number" value="8192"></div>
      </div>
      <button class="btn btn-primary" onclick="planPipeline()">🔀 Plan Pipeline</button>
      <div id="pipe-output" class="output-box"></div>
    </div>
    <div class="card">
      <h3>📊 Pipeline Status</h3>
      <div id="pipe-status">No pipeline configured</div>
    </div>
  </div>

  <!-- NETWORK -->
  <div id="network" class="panel">
    <div class="card">
      <h3>🌐 Network Status</h3>
      <div id="net-status">Loading...</div>
    </div>
    <div class="card">
      <h3>⚡ Inference Cluster</h3>
      <div id="net-cluster">Loading...</div>
    </div>
  </div>
</div>

<script>
let BASE = '';
const token = localStorage.getItem('netai_token') || '';

async function api(path, opts = {}) {
  const headers = {'Content-Type': 'application/json', ...opts.headers};
  if (token) headers['Authorization'] = 'Bearer ' + token;
  try {
    const r = await fetch(BASE + path, {...opts, headers});
    if (r.status === 401) return {error: 'Auth required. Run: netai auth register then login'};
    const ct = r.headers.get('content-type') || '';
    return ct.includes('json') ? r.json() : {status: r.status, text: await r.text().catch(()=>'')};
  } catch(e) { return {error: 'Connection failed: ' + e.message}; }
}

function toast(msg, type='info') {
  const el = document.createElement('div');
  el.style.cssText = 'padding:12px 20px;border-radius:6px;margin-bottom:8px;font-size:13px;font-weight:500;animation:fadeIn .3s;max-width:400px';
  el.style.background = type==='ok'?'#1b4332':type==='err'?'#4c1111':'#0c2d6b';
  el.style.color = type==='ok'?'#7ee787':type==='err'?'#f85149':'#58a6ff';
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

function showTab(id) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`.tab[onclick*="${id}"]`)?.classList.add('active');
  document.getElementById(id)?.classList.add('active');
  if (id === 'overview') loadOverview();
  if (id === 'models') loadModels();
  if (id === 'network') loadNetwork();
}

function errCheck(d) { if (d && d.error) { toast(d.error, 'err'); return true; } if (d && d.detail) { toast(d.detail, 'err'); return true; } return false; }

async function loadOverview() {
  const [status, native, resources] = await Promise.all([
    api('/api/status'),
    api('/api/inference/native/status'),
    api('/api/resources'),
  ]);
  const s = status || {};
  const n = native || {};
  const r = resources || {};
  document.getElementById('node-id').textContent = s.node_id || 'unknown';
  const p = s.profile || r;
  document.getElementById('ov-stats').innerHTML = `
    <div class="stat"><div class="value">${n.loaded_models?.length||0}</div><div class="label">Models Loaded</div></div>
    <div class="stat"><div class="value">${n.num_layers_loaded||0}</div><div class="label">Layers</div></div>
    <div class="stat"><div class="value">${n.weights_memory_mb?n.weights_memory_mb.toFixed(0):0}</div><div class="label">Memory (MB)</div></div>
    <div class="stat"><div class="value">${p.cpu_available||'?'}/${p.cpu_cores||'?'}</div><div class="label">CPU Cores</div></div>
    <div class="stat"><div class="value">${p.gpu_count||0}</div><div class="label">GPUs</div></div>
    <div class="stat"><div class="value">${(p.ram_available_gb||0).toFixed(1)}/${(p.ram_total_gb||0).toFixed(1)}</div><div class="label">RAM (GB)</div></div>
    <div class="stat"><div class="value">${s.peer_count||0}</div><div class="label">Peers</div></div>
  `;
  document.getElementById('ov-hardware').innerHTML = `<table>
    <tr><th style="width:200px">CPU</th><td>${p.cpu_available||'?'}/${p.cpu_cores||'?'} cores</td><td>${p.cpu_arch||''}</td></tr>
    <tr><th>GPU</th><td>${p.gpu_count||0} (${(p.gpu_names||[]).join(', ')||'none'})</td><td>VRAM: ${(p.gpu_available_vram_mb||[]).join('/')||0} MB</td></tr>
    <tr><th>CUDA</th><td>${p.has_cuda?'✅':'❌'}</td><td>ROCm: ${p.has_rocm?'✅':'❌'} | Vulkan: ${p.has_vulkan?'✅':'❌'}</td></tr>
    <tr><th>PyTorch</th><td>${p.torch_available?'✅ Available':'❌'}</td><td>Backends: ${(n.backends||[]).join(', ')||'numpy'}</td></tr>
    <tr><th>Capacity</th><td>Score: ${(p.capacity_score||0).toFixed(1)}</td><td>${p.summary||''}</td></tr>
  </table>`;
  document.getElementById('ov-native').innerHTML = `<table>
    <tr><th style="width:200px">Node ID</th><td>${n.node_id||'-'}</td><td></td></tr>
    <tr><th>Models</th><td>${(n.loaded_models||[]).join(', ')||'None — <a href="javascript:showTab(\\'download\\')">download a model</a>'}</td><td>${n.num_layers_loaded||0} layers</td></tr>
    <tr><th>Memory</th><td>${(n.weights_memory_mb||0).toFixed(1)} MB</td><td>Backends: ${(n.backends||[]).join(', ')}</td></tr>
  </table>`;
  const sel = document.getElementById('gen-model');
  if (sel && (n.loaded_models||[]).length > 0 && sel.options.length <= 1) {
    sel.innerHTML = '<option value="">-- Select a loaded model --</option>' +
      (n.loaded_models||[]).map(m => `<option value="${m}">${m}</option>`).join('');
  }
}

async function loadModels() {
  const [native, catalog] = await Promise.all([
    api('/api/inference/native/models'),
    api('/api/models/catalog'),
  ]);
  const n = native || {};
  const loaded = n.loaded_models || [];
  const cached = n.cached_models || [];
  const models = n.status || {};
  document.getElementById('models-loaded').innerHTML = loaded.length ?
    loaded.map(m => `<div class="model-card"><div class="model-name">${m}</div><div class="model-meta">Loaded in native engine | ${models.weights_memory_mb?.toFixed(0)||'?'} MB weights</div></div>`).join('') :
    '<p style="color:#8b949e">No models loaded. Download or load a model to start.</p>';
  document.getElementById('models-cached').innerHTML = cached.length ?
    cached.map(m => `<div class="model-card"><div class="model-name">${m}</div><div class="model-meta">Cached locally</div></div>`).join('') :
    '<p style="color:#8b949e">No cached models.</p>';
  const cat = catalog || {};
  const catModels = cat.models || [];
  document.getElementById('models-catalog').innerHTML = catModels.length ?
    `<table><tr><th>ID</th><th>Name</th><th>Class</th><th>Params</th><th>Min VRAM</th><th>License</th></tr>` +
    catModels.map(m => `<tr><td style="color:#58a6ff">${m.model_id}</td><td>${m.name}</td><td><span class="badge badge-info">${m.size_class}</span></td><td>${m.params_m?.toFixed(0)||'?'}M</td><td>${Math.min(...Object.values(m.vram_required_mb||{0:0}))||'?'} MB</td><td>${m.license||'-'}</td></tr>`).join('') + '</table>' :
    '<p style="color:#8b949e">No catalog models available.</p>';
  const sel = document.getElementById('gen-model');
  sel.innerHTML = '<option value="">-- Select a loaded model --</option>' +
    loaded.map(m => `<option value="${m}">${m}</option>`).join('');
}

async function generateNative() {
  const model = document.getElementById('gen-model').value;
  const prompt = document.getElementById('gen-prompt').value;
  if (!model) { toast('Select a model first', 'err'); return; }
  if (!prompt) { toast('Enter a prompt', 'err'); return; }
  const btn = document.getElementById('gen-btn');
  const out = document.getElementById('gen-output');
  const meta = document.getElementById('gen-meta');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Generating...';
  out.className = 'output-box';
  out.style.display = 'none';
  meta.style.display = 'none';
  const t0 = Date.now();
  const data = await api('/api/inference/native-run', {
    method: 'POST',
    headers: {'Authorization': 'Bearer ' + (localStorage.getItem('netai_token')||'')},
    body: JSON.stringify({model_id: model, prompt, max_tokens: parseInt(document.getElementById('gen-max-tokens').value)||64, temperature: parseFloat(document.getElementById('gen-temp').value)||0.7, top_p: parseFloat(document.getElementById('gen-topp').value)||0.9})
  });
  btn.disabled = false;
  btn.innerHTML = '⚡ Generate';
  if (errCheck(data)) return;
  const elapsed = ((Date.now()-t0)/1000).toFixed(2);
  out.className = 'output-box visible';
  out.style.display = 'block';
  if (data.error) {
    out.className = 'output-box visible error';
    out.textContent = 'Error: ' + data.error;
  } else {
    const genTokens = data.generated_tokens || [];
    const text = data.text || `[${genTokens.length} tokens generated]`;
    out.textContent = text;
    meta.style.display = 'flex';
    meta.innerHTML = `
      <span>⏱ ${data.latency_ms?.toFixed(0) || elapsed*1000}ms</span>
      <span>🚀 ${data.tokens_per_second?.toFixed(1) || (genTokens.length/(elapsed||1)).toFixed(1)} tok/s</span>
      <span>📝 ${genTokens.length || data.num_generated || 0} tokens</span>
      <span>🔢 Prompt: ${data.prompt_tokens?.length || 0} tokens</span>
      ${data.num_stages ? `<span>🔀 ${data.num_stages} stages</span>` : ''}
    `;
  }
}

async function loadLocalModel() {
  const dir = document.getElementById('ll-dir')?.value || '';
  const mid = document.getElementById('ll-modelid')?.value || '';
  if (!dir && !document.getElementById('gen-model').value) { toast('Enter directory path or select a model', 'err'); return; }
  const model = mid || document.getElementById('gen-model').value;
  const layerStart = parseInt(document.getElementById('ll-layer-start')?.value) || -1;
  const layerEnd = parseInt(document.getElementById('ll-layer-end')?.value) || -1;
  let url = `/api/inference/load-local?model_dir=${encodeURIComponent(dir)}&model_id=${encodeURIComponent(model)}`;
  if (layerStart >= 0) url += `&layer_start=${layerStart}`;
  if (layerEnd >= 0) url += `&layer_end=${layerEnd}`;
  const data = await api(url, {method: 'POST'});
  if (errCheck(data)) return;
  toast(`Model ${data.model_id} loaded: ${data.loaded_layers} layers, ${data.memory_mb?.toFixed(1)} MB`, 'ok');
  loadModels();
}

async function downloadModel() {
  const modelId = document.getElementById('dl-model').value.trim();
  const revision = document.getElementById('dl-revision').value.trim() || 'main';
  if (!modelId) { toast('Enter a model ID', 'err'); return; }
  const btn = document.getElementById('dl-btn');
  const out = document.getElementById('dl-output');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Downloading...';
  out.className = 'output-box';
  out.style.display = 'none';
  const data = await api(`/api/inference/download/${encodeURIComponent(modelId)}?revision=${encodeURIComponent(revision)}`, {method:'POST'});
  btn.disabled = false;
  btn.innerHTML = '📥 Download & Load';
  out.className = 'output-box visible';
  out.style.display = 'block';
  if (data.error) {
    out.className = 'output-box visible error';
    out.textContent = 'Error: ' + data.error;
    toast('Download failed: ' + data.error, 'err');
  } else {
    out.textContent = `✅ Model: ${data.model_id}\nLayers: ${data.loaded_layers}\nMemory: ${data.memory_mb?.toFixed(1)} MB\nFiles: ${data.files_downloaded}\nSize: ${data.total_size_mb?.toFixed(1)} MB\nVerified: ${data.verified}`;
    toast(`Downloaded ${data.model_id}: ${data.loaded_layers} layers`, 'ok');
    loadModels();
  }
}

async function planPipeline() {
  const model = document.getElementById('pipe-model').value.trim();
  const vram = parseInt(document.getElementById('pipe-vram').value) || 8192;
  if (!model) { toast('Enter model ID', 'err'); return; }
  const data = await api(`/api/inference/pipeline/plan?model_id=${encodeURIComponent(model)}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify([{node_id: 'local', vram_available_mb: vram}])
  });
  const out = document.getElementById('pipe-output');
  out.className = 'output-box visible';
  out.style.display = 'block';
  if (data.error || data.detail) {
    out.className = 'output-box visible error';
    out.textContent = 'Error: ' + (data.error || data.detail);
  } else {
    let html = `Model: ${data.model_id}\nTotal stages: ${data.total_stages}\n\n`;
    for (const s of data.stages || []) {
      html += `Stage ${s.stage_index}: layers ${s.layer_start}-${s.layer_end} (${s.num_layers} layers) | ${s.vram_mb?.toFixed(0)}MB | ${s.node_id} | ${s.status}\n`;
    }
    out.textContent = html;
  }
  const ps = await api(`/api/inference/pipeline/status?model_id=${encodeURIComponent(model)}`);
  document.getElementById('pipe-status').innerHTML = ps.error ?
    `<p style="color:#8b949e">${ps.error}</p>` :
    `<pre style="color:#7ee787;font-family:monospace;font-size:13px">${JSON.stringify(ps, null, 2)}</pre>`;
}

async function loadNetwork() {
  const [status, inf] = await Promise.all([api('/api/status'), api('/api/inference/status')]);
  const s = status || {};
  const i = inf || {};
  const local = i.local || {};
  const cluster = i.cluster || {};
  document.getElementById('net-status').innerHTML = `<table>
    <tr><th style="width:200px">Node</th><td>${s.node_id||'-'}</td><td><span class="badge badge-info">${s.state||'-'}</span></td></tr>
    <tr><th>Peers</th><td>${s.peer_count||0}</td><td></td></tr>
    <tr><th>Jobs</th><td>${(s.jobs||[]).length}</td><td></td></tr>
    <tr><th>Groups</th><td>${s.groups||0}</td><td></td></tr>
    <tr><th>Proposals</th><td>${s.proposals||0}</td><td></td></tr>
  </table>`;
  document.getElementById('net-cluster').innerHTML = `<table>
    <tr><th style="width:200px">Strategy</th><td>${cluster.strategy||'-'}</td><td></td></tr>
    <tr><th>Total Nodes</th><td>${cluster.total_nodes||0}</td><td>Available: ${cluster.available_nodes||0}</td></tr>
    <tr><th>Models</th><td>${local.models_loaded||0}</td><td>Total inferences: ${local.total_inferences||0}</td></tr>
    <tr><th>Inferences</th><td>${local.total_inferences||0}</td><td></td></tr>
  </table>`;
}

loadOverview();
loadModels();
</script>
</body>
</html>"""


def _ws_test_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NetAI WebSocket Test</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:#0a0a0f;color:#e0e0e0;min-height:100vh;padding:32px}
h1{color:#58a6ff;font-size:24px;margin-bottom:8px}
.subtitle{color:#8b949e;font-size:13px;margin-bottom:24px}
.card{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:20px;margin-bottom:16px;max-width:800px}
.card h3{color:#58a6ff;font-size:15px;margin-bottom:12px}
.form-group{margin-bottom:12px}
.form-group label{display:block;font-size:11px;color:#8b949e;margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px;font-weight:600}
.form-group input,.form-group textarea{width:100%;padding:8px 12px;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e0e0e0;font-size:14px;font-family:inherit}
.form-group input:focus,.form-group textarea:focus{outline:none;border-color:#58a6ff;box-shadow:0 0 0 1px #1f6feb33}
.form-row{display:flex;gap:12px;flex-wrap:wrap}
.form-row .form-group{flex:1;min-width:140px}
.btn{padding:8px 16px;border:1px solid #30363d;background:#21262d;color:#c9d1d9;border-radius:6px;cursor:pointer;font-size:13px;font-weight:500;margin-right:8px}
.btn:hover{background:#30363d;border-color:#58a6ff;color:#fff}
.btn-primary{background:#1f6feb;border-color:#1f6feb;color:#fff}
.btn-primary:hover{background:#388bfd}
.btn-danger{background:#da3633;border-color:#da3633;color:#fff;margin-top:12px}
.output-box{margin-top:12px;padding:16px;background:#0d1117;border:1px solid #21262d;border-radius:6px;font-family:'JetBrains Mono',monospace;font-size:13px;white-space:pre-wrap;word-break:break-all;max-height:400px;overflow-y:auto;color:#7ee787;min-height:80px}
.connected{color:#7ee787;font-weight:600}
.disconnected{color:#f85149;font-weight:600}
.status-bar{display:flex;align-items:center;gap:12px;margin-bottom:12px;font-size:13px}
</style>
</head>
<body>
<h1>WebSocket Native Streaming Test</h1>
<p class="subtitle">Test token-by-token streaming via WebSocket NativeInferenceEngine</p>

<div class="card">
  <h3>Connection</h3>
  <div class="status-bar">
    <span>Status: </span>
    <span id="conn-status" class="disconnected">Not Connected</span>
  </div>
  <div class="form-row">
    <div class="form-group"><label>Model ID</label><input id="model-id" value="gpt2"></div>
  </div>
  <button class="btn btn-primary" onclick="connect()">Connect</button>
  <button class="btn" onclick="disconnectWS()">Disconnect</button>
</div>

<div class="card">
  <h3>Inference</h3>
  <div class="form-row">
    <div class="form-group"><label>Prompt</label><textarea id="prompt" rows="3">The meaning of life is</textarea></div>
  </div>
  <div class="form-row">
    <div class="form-group"><label>Max Tokens</label><input id="max-tokens" type="number" value="32" min="1" max="128"></div>
    <div class="form-group"><label>Temperature</label><input id="temperature" type="number" value="0.7" min="0" max="2" step="0.1"></div>
    <div class="form-group"><label>Top-P</label><input id="top-p" type="number" value="0.9" min="0" max="1" step="0.05"></div>
  </div>
  <button class="btn btn-primary" onclick="sendInference()" id="send-btn" disabled>Send</button>
  <button class="btn btn-danger" onclick="sendCancel()" id="cancel-btn" disabled>Cancel</button>
  <div class="output-box" id="output"></div>
</div>

<script>
let ws = null;
let streaming = false;

function connect() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    disconnectWS();
  }
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = proto + '//' + location.host + '/ws/inference/stream-native';
  ws = new WebSocket(url);
  
  ws.onopen = () => {
    document.getElementById('conn-status').textContent = 'Connected';
    document.getElementById('conn-status').className = 'connected';
    document.getElementById('send-btn').disabled = false;
  };
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const out = document.getElementById('output');
    if (data.type === 'start') {
      out.textContent = '';
      document.getElementById('cancel-btn').disabled = false;
      streaming = true;
    } else if (data.type === 'token') {
      out.textContent += data.text;
    } else if (data.type === 'done') {
      out.textContent += '\\n\\n---\\nGenerated ' + data.tokens_generated + ' tokens in ' + data.latency_ms + 'ms (' + data.tokens_per_second + ' tok/s)';
      document.getElementById('cancel-btn').disabled = true;
      streaming = false;
    } else if (data.type === 'cancelled') {
      out.textContent += '\\n\\n[CANCELLED at ' + data.tokens_generated + ' tokens]';
      document.getElementById('cancel-btn').disabled = true;
      streaming = false;
    } else if (data.type === 'error') {
      out.textContent = 'ERROR: ' + data.error;
      document.getElementById('cancel-btn').disabled = true;
      streaming = false;
    }
  };
  
  ws.onclose = () => {
    document.getElementById('conn-status').textContent = 'Disconnected';
    document.getElementById('conn-status').className = 'disconnected';
    document.getElementById('send-btn').disabled = true;
    document.getElementById('cancel-btn').disabled = true;
    streaming = false;
    ws = null;
  };
  
  ws.onerror = () => {
    document.getElementById('output').textContent = 'WebSocket connection error';
  };
}

function disconnectWS() {
  if (ws) {
    ws.close();
    ws = null;
  }
}

function sendInference() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const msg = {
    model_id: document.getElementById('model-id').value,
    prompt: document.getElementById('prompt').value,
    max_tokens: parseInt(document.getElementById('max-tokens').value) || 32,
    temperature: parseFloat(document.getElementById('temperature').value) || 0.7,
    top_p: parseFloat(document.getElementById('top-p').value) || 0.9,
  };
  ws.send(JSON.stringify(msg));
}

function sendCancel() {
  if (!ws || ws.readyState !== WebSocket.OPEN || !streaming) return;
  ws.send(JSON.stringify({type: 'cancel'}));
}
</script>
</body>
</html>"""
