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
from netai.training.engine import GradientSyncServer
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
) -> FastAPI:
    p2p = p2p_node or P2PNode()
    coord = coordinator or DistributedTrainingCoordinator(p2p)
    voting = voting_engine or VotingEngine()
    groups = group_manager or GroupManager()
    gh = github or GitHubIntegration()
    sched = scheduler or JobScheduler()
    inf_engine = InferenceEngine(node_id=p2p.node_id)
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
            return {"error": "Inference gateway error", "request_id": request.request_id}

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

    return app


def _dashboard_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NetAI - Distributed AI Training</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; }
  .header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); padding: 24px 32px; border-bottom: 1px solid #1a3a5c; }
  .header h1 { font-size: 28px; color: #00d2ff; }
  .header p { color: #8899aa; margin-top: 4px; font-size: 14px; }
  .tabs { display: flex; background: #111; border-bottom: 1px solid #333; overflow-x: auto; }
  .tab { padding: 12px 24px; cursor: pointer; border-bottom: 2px solid transparent; color: #888; font-size: 14px; white-space: nowrap; transition: all 0.2s; }
  .tab:hover { color: #ccc; }
  .tab.active { color: #00d2ff; border-bottom-color: #00d2ff; }
  .content { padding: 24px 32px; max-width: 1400px; margin: 0 auto; }
  .panel { display: none; }
  .panel.active { display: block; }
  .card { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px; margin-bottom: 16px; }
  .card h3 { color: #00d2ff; margin-bottom: 12px; font-size: 16px; }
  .stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .stat { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 16px; text-align: center; }
  .stat .value { font-size: 32px; font-weight: bold; color: #00d2ff; }
  .stat .label { font-size: 12px; color: #888; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #2a2a2a; font-size: 13px; }
  th { color: #888; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 1px; }
  tr:hover { background: #222; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .badge.running { background: #0066cc33; color: #4da6ff; }
  .badge.completed { background: #00cc6633; color: #00cc66; }
  .badge.pending { background: #cccc0033; color: #cccc00; }
  .badge.failed { background: #cc003333; color: #cc3333; }
  .badge.active { background: #0066cc33; color: #4da6ff; }
  .badge.passed { background: #00cc6633; color: #00cc66; }
  .btn { padding: 8px 16px; border: 1px solid #333; background: #222; color: #e0e0e0; border-radius: 6px; cursor: pointer; font-size: 13px; }
  .btn:hover { background: #333; border-color: #00d2ff; }
  .btn-primary { background: #004466; border-color: #00d2ff; color: #00d2ff; }
  .btn-primary:hover { background: #006688; }
  .form-group { margin-bottom: 12px; }
  .form-group label { display: block; font-size: 12px; color: #888; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1px; }
  .form-group input, .form-group select, .form-group textarea { width: 100%; padding: 8px 12px; background: #111; border: 1px solid #333; border-radius: 6px; color: #e0e0e0; font-size: 14px; }
  .form-group input:focus, .form-group select:focus { outline: none; border-color: #00d2ff; }
  .form-row { display: flex; gap: 12px; flex-wrap: wrap; }
  .form-row .form-group { flex: 1; min-width: 150px; }
  .progress-bar { height: 8px; background: #2a2a2a; border-radius: 4px; overflow: hidden; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, #00d2ff, #00ff88); border-radius: 4px; transition: width 0.3s; }
  .vote-bar { display: flex; height: 24px; border-radius: 4px; overflow: hidden; margin: 8px 0; }
  .vote-for { background: #00cc66; display: flex; align-items: center; justify-content: center; font-size: 11px; color: #000; font-weight: 600; }
  .vote-against { background: #cc3333; display: flex; align-items: center; justify-content: center; font-size: 11px; color: #fff; font-weight: 600; }
  #toast-container { position: fixed; top: 20px; right: 20px; z-index: 9999; }
  .toast { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; font-size: 13px; animation: slideIn 0.3s; }
  @keyframes slideIn { from { transform: translateX(100px); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
</style>
</head>
<body>
<div class="header">
  <h1>NetAI</h1>
  <p>Open-Source Distributed AI Training &bull; P2P Resource Pooling &bull; CPU/GPU Training &bull; Governance Voting</p>
</div>
<div class="tabs">
  <div class="tab active" onclick="showTab('overview')">Overview</div>
  <div class="tab" onclick="showTab('training')">Training</div>
  <div class="tab" onclick="showTab('voting')">Voting</div>
  <div class="tab" onclick="showTab('groups')">Groups</div>
  <div class="tab" onclick="showTab('pledges')">Resources</div>
  <div class="tab" onclick="showTab('peers')">Peers</div>
  <div class="tab" onclick="showTab('inference')">Inference</div>
  <div class="tab" onclick="showTab('jackin')">Jack In</div>
  <div class="tab" onclick="showTab('github')">GitHub</div>
  <div class="tab" onclick="showTab('gradients')">Gradients</div>
  <div class="tab" onclick="showTab('metrics')">Metrics</div>
  <div class="tab" onclick="showTab('auth')">Auth</div>
</div>
<div class="content">
  <div id="toast-container"></div>

  <div id="overview" class="panel active">
    <div class="stats" id="overview-stats"></div>
    <div class="card"><h3>System Status</h3><div id="sys-status">Loading...</div></div>
    <div class="card"><h3>Recent Activity</h3><div id="recent-activity">Loading...</div></div>
  </div>

  <div id="training" class="panel">
    <div class="card">
      <h3>Submit Training Job</h3>
      <div class="form-row">
        <div class="form-group"><label>Model Name</label><input id="t-model" value="gpt2-small"></div>
        <div class="form-group"><label>Architecture</label><select id="t-arch"><option>transformer</option><option>mlp</option><option>cnn</option></select></div>
        <div class="form-group"><label>Hidden Size</label><input id="t-hidden" type="number" value="768"></div>
        <div class="form-group"><label>Layers</label><input id="t-layers" type="number" value="12"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Steps</label><input id="t-steps" type="number" value="1000"></div>
        <div class="form-group"><label>Batch Size</label><input id="t-batch" type="number" value="8"></div>
        <div class="form-group"><label>Learning Rate</label><input id="t-lr" value="3e-4"></div>
        <div class="form-group"><label>Device</label><select id="t-device"><option value="cuda">CUDA GPU</option><option value="cpu">CPU</option><option value="rocm">ROCm</option><option value="mps">MPS</option></select></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Group ID (optional)</label><input id="t-group" placeholder="Leave empty for public"></div>
      </div>
      <button class="btn btn-primary" onclick="submitTraining()">Submit Training Job</button>
    </div>
    <div class="card"><h3>Active Jobs</h3><div id="jobs-list">Loading...</div></div>
  </div>

  <div id="voting" class="panel">
    <div class="card">
      <h3>Propose a Model to Train</h3>
      <div class="form-row">
        <div class="form-group"><label>Model Name</label><input id="v-model" value="llama-7b-finetune"></div>
        <div class="form-group"><label>Architecture</label><select id="v-arch"><option>transformer</option><option>mamba</option><option>rwkv</option></select></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Description</label><textarea id="v-desc" rows="2">Fine-tune LLaMA 7B on domain-specific data</textarea></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Your ID</label><input id="v-proposer" value="user-1"></div>
        <div class="form-group"><label>Group (optional)</label><input id="v-group" placeholder="Any group"></div>
      </div>
      <button class="btn btn-primary" onclick="proposeModel()">Submit Proposal</button>
    </div>
    <div class="card"><h3>Active Proposals</h3><div id="proposals-list">Loading...</div></div>
    <div class="card">
      <h3>Cast Vote</h3>
      <div class="form-row">
        <div class="form-group"><label>Proposal ID</label><input id="cv-proposal" placeholder="proposal-id"></div>
        <div class="form-group"><label>Voter ID</label><input id="cv-voter" value="user-1"></div>
        <div class="form-group"><label>Vote</label><select id="cv-choice"><option value="for">For</option><option value="against">Against</option></select></div>
      </div>
      <button class="btn btn-primary" onclick="castVote()">Cast Vote</button>
    </div>
  </div>

  <div id="groups" class="panel">
    <div class="card">
      <h3>Create Training Group</h3>
      <div class="form-row">
        <div class="form-group"><label>Group Name</label><input id="g-name" value="my-research-team"></div>
        <div class="form-group"><label>Owner ID</label><input id="g-owner" value="user-1"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Description</label><textarea id="g-desc" rows="2"></textarea></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Visibility</label><select id="g-vis"><option value="private">Private</option><option value="public">Public</option><option value="secret">Secret</option></select></div>
        <div class="form-group"><label>Max Members</label><input id="g-max" type="number" value="50"></div>
        <div class="form-group"><label>Require Approval</label><select id="g-approval"><option value="true">Yes</option><option value="false">No</option></select></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Passphrase (optional)</label><input id="g-pass" type="password" placeholder="Leave empty for auto-key"></div>
      </div>
      <button class="btn btn-primary" onclick="createGroup()">Create Group</button>
    </div>
    <div class="card">
      <h3>Join a Group</h3>
      <div class="form-row">
        <div class="form-group"><label>Group ID</label><input id="jg-id" placeholder="group-id"></div>
        <div class="form-group"><label>User ID</label><input id="jg-user" value="user-1"></div>
        <div class="form-group"><label>Invite Code</label><input id="jg-invite" placeholder="Optional for public groups"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>CPU Cores</label><input id="jg-cpu" type="number" value="4"></div>
        <div class="form-group"><label>GPU Count</label><input id="jg-gpu" type="number" value="0"></div>
        <div class="form-group"><label>RAM (GB)</label><input id="jg-ram" type="number" value="16"></div>
      </div>
      <button class="btn btn-primary" onclick="joinGroup()">Join Group</button>
    </div>
    <div class="card"><h3>All Groups</h3><div id="groups-list">Loading...</div></div>
  </div>

  <div id="pledges" class="panel">
    <div class="card">
      <h3>Pledge Resources</h3>
      <div class="form-row">
        <div class="form-group"><label>User ID</label><input id="p-user" value="user-1"></div>
        <div class="form-group"><label>Node ID</label><input id="p-node" placeholder="auto"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>CPU Cores</label><input id="p-cpu" type="number" value="4"></div>
        <div class="form-group"><label>GPU Count</label><input id="p-gpu" type="number" value="1"></div>
        <div class="form-group"><label>RAM (GB)</label><input id="p-ram" type="number" value="32"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Time (hours)</label><input id="p-time" type="number" value="24"></div>
        <div class="form-group"><label>Group (optional)</label><input id="p-group" placeholder="Anyone can use"></div>
      </div>
      <button class="btn btn-primary" onclick="pledgeResources()">Pledge Resources</button>
    </div>
    <div class="card"><h3>Cluster Resources</h3><div id="cluster-resources">Loading...</div></div>
    <div class="card"><h3>Leaderboard</h3><div id="leaderboard">Loading...</div></div>
  </div>

  <div id="peers" class="panel">
    <div class="card"><h3>Peer Network</h3><div id="peers-list">Loading...</div></div>
  </div>

  <div id="inference" class="panel">
    <div class="card">
      <h3>Load Model for Inference</h3>
      <div class="form-row">
        <div class="form-group"><label>Model ID</label><input id="inf-model" value="gpt2-small"></div>
        <div class="form-group"><label>Version</label><input id="inf-version" value="latest"></div>
        <div class="form-group"><label>Shards</label><input id="inf-shards" type="number" value="1"></div>
        <div class="form-group"><label>Device</label><select id="inf-device"><option value="auto">Auto</option><option value="cuda">CUDA</option><option value="cpu">CPU</option></select></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Mirror Enabled</label><select id="inf-mirror"><option value="true">Yes</option><option value="false">No</option></select></div>
        <div class="form-group"><label>Group (optional)</label><input id="inf-group" placeholder=""></div>
      </div>
      <button class="btn btn-primary" onclick="loadModel()">Load Model</button>
    </div>
    <div class="card">
      <h3>Run Inference</h3>
      <div class="form-row">
        <div class="form-group"><label>Model</label><input id="inf-run-model" value="gpt2-small"></div>
        <div class="form-group"><label>Max Tokens</label><input id="inf-max-tokens" type="number" value="128"></div>
        <div class="form-group"><label>Temperature</label><input id="inf-temp" type="number" value="0.7" step="0.1"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Prompt</label><textarea id="inf-prompt" rows="3">The meaning of life is</textarea></div>
      </div>
      <button class="btn btn-primary" onclick="runInference()">Generate</button>
      <div id="inf-output" style="margin-top:12px;padding:12px;background:#111;border:1px solid #333;border-radius:6px;display:none;"></div>
    </div>
    <div class="card"><h3>Inference Cluster Status</h3><div id="inf-status">Loading...</div></div>
  </div>

  <div id="jackin" class="panel">
    <div class="card">
      <h3>&#9889; Jack In &mdash; Join the Network</h3>
      <p style="color:#888;margin-bottom:16px">Jack into distributed training or inference. Your resources (CPU/GPU/RAM) are pooled and you earn reputation for contributing.</p>
      <div class="form-row">
        <div class="form-group"><label>User ID</label><input id="ji-user" value="user-1"></div>
        <div class="form-group"><label>Mode</label><select id="ji-mode"><option value="both">Training + Inference</option><option value="training">Training Only</option><option value="inference">Inference Only</option></select></div>
        <div class="form-group"><label>Group (optional)</label><input id="ji-group" placeholder="Any group"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>CPU Cores (0=auto)</label><input id="ji-cpu" type="number" value="0"></div>
        <div class="form-group"><label>GPU Count (0=auto)</label><input id="ji-gpu" type="number" value="0"></div>
        <div class="form-group"><label>RAM GB (0=auto)</label><input id="ji-ram" type="number" value="0"></div>
        <div class="form-group"><label>Hours</label><input id="ji-hours" type="number" value="24"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Models to Serve (comma-separated)</label><input id="ji-models" placeholder="gpt2-small,llama-7b"></div>
      </div>
      <button class="btn btn-primary" onclick="jackIn()">&#9889; Jack In</button>
      <div id="ji-result" style="margin-top:12px;display:none;"></div>
    </div>
    <div class="card"><h3>Current Network Resources</h3><div id="ji-network">Loading...</div></div>
  </div>

  <div id="github" class="panel">
    <div class="card">
      <h3>GitHub Integration</h3>
      <p style="color:#888;margin-bottom:16px">Configure webhook to your repo to auto-trigger training on commits. POST to <code>/api/github/webhook</code> with X-Hub-Signature-256 header.</p>
      <div class="form-row">
        <div class="form-group"><label>Repo URL</label><input id="gh-repo" placeholder="https://github.com/user/repo"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Branch</label><input id="gh-branch" value="main"></div>
        <div class="form-group"><label>Webhook Secret</label><input id="gh-secret" type="password"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Access Token</label><input id="gh-token" type="password"></div>
      </div>
    </div>
    <div class="card"><h3>Commit-Triggered Training</h3><p style="color:#888">When enabled, pushing to the configured branch triggers automatic training. Config file: <code>netai.yaml</code> in repo root.</p></div>
  </div>

  <div id="gradients" class="panel">
    <div class="card">
      <h3>Gradient Sync</h3>
      <p style="color:#888;margin-bottom:16px">Exchange gradient updates between P2P nodes for distributed training.</p>
      <div class="form-row">
        <div class="form-group"><label>Job ID</label><input id="gs-job" placeholder="job-id"></div>
        <div class="form-group"><label>Step</label><input id="gs-step" type="number" value="1"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Peer Node ID</label><input id="gs-peer-id" placeholder="node-id"></div>
        <div class="form-group"><label>Peer Endpoint</label><input id="gs-peer-ep" placeholder="http://peer:8000"></div>
      </div>
      <button class="btn btn-primary" onclick="addGradientPeer()">Add Peer</button>
      <button class="btn" onclick="aggregateGradients()" style="margin-left:8px">Aggregate</button>
      <button class="btn" onclick="pushGradients()" style="margin-left:8px">Push</button>
      <button class="btn" onclick="pullGradients()" style="margin-left:8px">Pull</button>
    </div>
    <div class="card"><h3>Sync Status</h3><div id="gs-status">Loading...</div></div>
  </div>

  <div id="metrics" class="panel">
    <div class="card">
      <h3>Prometheus Metrics</h3>
      <p style="color:#888;margin-bottom:16px">Export metrics in Prometheus format. Scrape endpoint: <code>/api/metrics</code></p>
      <button class="btn btn-primary" onclick="loadMetrics()">Refresh Metrics</button>
    </div>
    <div class="card"><h3>Current Metrics</h3><pre id="metrics-content" style="color:#00ff88;font-family:monospace;font-size:12px;overflow-x:auto;white-space:pre-wrap;">Loading...</pre></div>
  </div>

  <div id="auth" class="panel">
    <div class="card">
      <h3>Create Auth Token</h3>
      <div class="form-row">
        <div class="form-group"><label>User ID</label><input id="at-user" value="user-1"></div>
        <div class="form-group"><label>TTL (hours)</label><input id="at-ttl" type="number" value="24"></div>
      </div>
      <button class="btn btn-primary" onclick="createAuthToken()">Generate Token</button>
      <div id="at-result" style="margin-top:12px;display:none;"></div>
    </div>
    <div class="card">
      <h3>Create API Key</h3>
      <div class="form-row">
        <div class="form-group"><label>User ID</label><input id="ak-user" value="user-1"></div>
        <div class="form-group"><label>Key Name</label><input id="ak-name" value="default"></div>
      </div>
      <button class="btn btn-primary" onclick="createApiKey()">Generate Key</button>
      <div id="ak-result" style="margin-top:12px;display:none;"></div>
    </div>
    <div class="card">
      <h3>Verify Token</h3>
      <div class="form-row">
        <div class="form-group"><label>Token</label><input id="vt-token" placeholder="Paste token here"></div>
      </div>
      <button class="btn" onclick="verifyToken()">Verify</button>
      <div id="vt-result" style="margin-top:12px;display:none;"></div>
    </div>
  </div>
</div>

<script>
let refreshInterval;

function showTab(id) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById(id).classList.add('active');
  loadPanel(id);
}

function toast(msg, type='info') {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = 'toast';
  t.style.borderLeftColor = type === 'ok' ? '#00cc66' : type === 'err' ? '#cc3333' : '#00d2ff';
  t.style.borderLeft = '3px solid';
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

async function api(path, opts={}) {
  try {
    const r = await fetch(path, opts);
    return await r.json();
  } catch(e) { toast('API error: '+e, 'err'); return null; }
}

async function loadPanel(id) {
  if (id === 'overview') {
    const s = await api('/api/status');
    if (!s) return;
    document.getElementById('overview-stats').innerHTML = `
      <div class="stat"><div class="value">${s.peer_count||0}</div><div class="label">Peers</div></div>
      <div class="stat"><div class="value">${(s.jobs||[]).length}</div><div class="label">Jobs</div></div>
      <div class="stat"><div class="value">${s.groups||0}</div><div class="label">Groups</div></div>
      <div class="stat"><div class="value">${s.proposals||0}</div><div class="label">Proposals</div></div>
      <div class="stat"><div class="value">${s.pledges||0}</div><div class="label">Pledges</div></div>
    `;
    const p = s.profile || {};
    document.getElementById('sys-status').innerHTML = `
      <table><tr><th>Property</th><th>Value</th></tr>
      <tr><td>Node ID</td><td>${s.node_id||'-'}</td></tr>
      <tr><td>CPU</td><td>${p.cpu_available||0}/${p.cpu_cores||0} cores</td></tr>
      <tr><td>GPU</td><td>${p.gpu_count||0} (${(p.gpu_names||[]).join(', ')||'none'})</td></tr>
      <tr><td>RAM</td><td>${(p.ram_available_gb||0).toFixed(1)}/${(p.ram_total_gb||0).toFixed(1)} GB</td></tr>
      <tr><td>CUDA</td><td>${p.has_cuda?'Yes':'No'}</td></tr>
      <tr><td>ROCm</td><td>${p.has_rocm?'Yes':'No'}</td></tr>
      <tr><td>Vulkan</td><td>${p.has_vulkan?'Yes':'No'}</td></tr>
      <tr><td>PyTorch</td><td>${p.torch_available?'Yes':'No'}</td></tr>
      </table>
    `;
  } else if (id === 'training') {
    const d = await api('/api/training/jobs');
    if (!d) return;
    const jobs = d.jobs || [];
    document.getElementById('jobs-list').innerHTML = jobs.length === 0 ? '<p style="color:#888">No jobs yet</p>' :
      '<table><tr><th>Job ID</th><th>Model</th><th>Status</th><th>Step</th><th>Loss</th><th>Elapsed</th></tr>' +
      jobs.map(j => `<tr><td>${j.job_id}</td><td>${(j.config||{}).model_name||'-'}</td>
      <td><span class="badge ${j.status}">${j.status}</span></td>
      <td>${j.step||0}</td><td>${j.loss!=null?j.loss.toFixed(4):'-'}</td>
      <td>${j.elapsed_s ? Math.round(j.elapsed_s)+'s':'-'}</td></tr>`).join('') +
      '</table>';
  } else if (id === 'voting') {
    const d = await api('/api/vote/proposals');
    if (!d) return;
    const ps = d.proposals || [];
    document.getElementById('proposals-list').innerHTML = ps.length === 0 ? '<p style="color:#888">No proposals yet</p>' :
      ps.map(p => {
        const total = p.weighted_for + p.weighted_against;
        const forPct = total > 0 ? (p.weighted_for/total*100).toFixed(0) : 50;
        return `<div class="card"><h3>${p.title} <span class="badge ${p.status}">${p.status}</span></h3>
        <p style="color:#888;font-size:12px">by ${p.proposer_id} &bull; Type: ${p.vote_type} &bull; ID: ${p.proposal_id}</p>
        <div class="vote-bar"><div class="vote-for" style="width:${forPct}%">${p.weighted_for.toFixed(1)} for</div>
        <div class="vote-against" style="width:${100-forPct}%">${p.weighted_against.toFixed(1)} against</div></div>
        <p style="font-size:12px;color:#888">${p.votes_for} votes for, ${p.votes_against} against | Quorum: ${p.quorum} | Threshold: ${p.threshold}</p></div>`;
      }).join('');
  } else if (id === 'groups') {
    const d = await api('/api/groups');
    if (!d) return;
    const gs = d.groups || [];
    document.getElementById('groups-list').innerHTML = gs.length === 0 ? '<p style="color:#888">No groups yet</p>' :
      '<table><tr><th>ID</th><th>Name</th><th>Visibility</th><th>Members</th><th>Tags</th></tr>' +
      gs.map(g => `<tr><td>${g.group_id}</td><td>${g.name}</td>
      <td><span class="badge active">${g.visibility}</span></td>
      <td>${g.members}</td><td>${(g.tags||[]).join(', ')||'-'}</td></tr>`).join('') + '</table>';
  } else if (id === 'pledges') {
    const [cr, lb] = await Promise.all([api('/api/resources/cluster'), api('/api/pledge/leaderboard')]);
    if (cr) document.getElementById('cluster-resources').innerHTML =
      `<div class="stats"><div class="stat"><div class="value">${cr.total_cpu_cores||0}</div><div class="label">CPU Cores</div></div>
      <div class="stat"><div class="value">${cr.total_gpu_count||0}</div><div class="label">GPUs</div></div>
      <div class="stat"><div class="value">${(cr.total_ram_gb||0).toFixed(0)}</div><div class="label">RAM GB</div></div>
      <div class="stat"><div class="value">${cr.total_vram_mb||0}</div><div class="label">VRAM MB</div></div>
      <div class="stat"><div class="value">${cr.num_contributors||0}</div><div class="label">Contributors</div></div></div>`;
    if (lb) document.getElementById('leaderboard').innerHTML =
      (lb.leaderboard||[]).length === 0 ? '<p style="color:#888">No pledges yet</p>' :
      '<table><tr><th>Rank</th><th>User</th><th>Score</th><th>CPU</th><th>GPU</th><th>RAM</th><th>Reputation</th></tr>' +
      lb.leaderboard.map(l => `<tr><td>#${l.rank}</td><td>${l.user_id}</td><td>${l.score.toFixed(1)}</td>
      <td>${l.cpu}</td><td>${l.gpu}</td><td>${l.ram_gb.toFixed(0)}</td><td>${l.reputation.toFixed(1)}</td></tr>`).join('') + '</table>';
  } else if (id === 'peers') {
    const d = await api('/api/peers');
    if (!d) return;
    document.getElementById('peers-list').innerHTML =
      (d.peers||[]).length === 0 ? '<p style="color:#888">No peers connected</p>' :
      '<table><tr><th>Node ID</th><th>Host</th><th>State</th><th>CPU</th><th>GPU</th><th>RAM</th></tr>' +
      d.peers.map(p => `<tr><td>${p.node_id}</td><td>${p.host}:${p.port}</td>
      <td><span class="badge ${p.state}">${p.state}</span></td>
      <td>${p.cpu_avail}/${p.cpu_cores}</td><td>${p.gpu_avail}/${p.gpu_count}</td>
       <td>${(p.ram_avail_gb||0).toFixed(1)}GB</td></tr>`).join('') + '</table>';
  } else if (id === 'inference') {
    const d = await api('/api/inference/status');
    if (d) {
      const local = d.local || {};
      const cluster = d.cluster || {};
      document.getElementById('inf-status').innerHTML =
        `<div class="stats"><div class="stat"><div class="value">${local.models_loaded||0}</div><div class="label">Local Models</div></div>
        <div class="stat"><div class="value">${(cluster.total_inferences||0)}</div><div class="label">Total Inferences</div></div>
        <div class="stat"><div class="value">${(cluster.available_nodes||0)}</div><div class="label">Available Nodes</div></div></div>`;
    }
  } else if (id === 'jackin') {
    const [sr, cr] = await Promise.all([api('/api/status'), api('/api/resources/cluster')]);
    if (sr && cr) {
      const p = sr.profile || {};
      document.getElementById('ji-network').innerHTML =
        `<div class="stats"><div class="stat"><div class="value">${p.cpu_available||0}</div><div class="label">CPU Available</div></div>
        <div class="stat"><div class="value">${p.gpu_available||0}</div><div class="label">GPU Available</div></div>
        <div class="stat"><div class="value">${(p.ram_available_gb||0).toFixed(1)}GB</div><div class="label">RAM Available</div></div>
        <div class="stat"><div class="value">${cr.total_cpu_cores||0}</div><div class="label">Cluster CPU</div></div>
        <div class="stat"><div class="value">${cr.total_gpu_count||0}</div><div class="label">Cluster GPU</div></div></div>`;
    }
  } else if (id === 'gradients') {
    const r = await api('/api/training/gradient-status');
    if (r) {
      let html = `<div class="stats"><div class="stat"><div class="value">${r.peers||0}</div><div class="label">Sync Peers</div></div>`;
      const jobs = Object.keys(r.gradient_store || {});
      html += `<div class="stat"><div class="value">${jobs.length}</div><div class="label">Jobs with Gradients</div></div>`;
      html += `<div class="stat"><div class="value">${r.running?'Yes':'No'}</div><div class="label">Running</div></div></div>`;
      if (jobs.length > 0) {
        html += '<table><tr><th>Job</th><th>Steps</th><th>Latest Agg. Step</th></tr>';
        for (const [jid, info] of Object.entries(r.gradient_store)) {
          html += `<tr><td>${jid}</td><td>${(info.steps_available||[]).join(', ')}</td><td>${info.aggregated_step||'-'}</td></tr>`;
        }
        html += '</table>';
      } else {
        html += '<p style="color:#888">No gradient data yet</p>';
      }
      document.getElementById('gs-status').innerHTML = html;
    }
  } else if (id === 'metrics') {
    const r = await fetch('/api/metrics');
    const text = await r.text();
    document.getElementById('metrics-content').textContent = text;
  }
}

async function submitTraining() {
  const body = {
    model_name: document.getElementById('t-model').value,
    model_architecture: document.getElementById('t-arch').value,
    hidden_size: +document.getElementById('t-hidden').value,
    num_layers: +document.getElementById('t-layers').value,
    total_steps: +document.getElementById('t-steps').value,
    batch_size: +document.getElementById('t-batch').value,
    learning_rate: +document.getElementById('t-lr').value,
    device_preference: document.getElementById('t-device').value,
    group_id: document.getElementById('t-group').value,
  };
  const r = await api('/api/training/submit', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  if (r) { toast('Job submitted: '+r.job_id, 'ok'); loadPanel('training'); }
}

async function proposeModel() {
  const body = {
    model_name: document.getElementById('v-model').value,
    architecture: document.getElementById('v-arch').value,
    description: document.getElementById('v-desc').value,
    proposer_id: document.getElementById('v-proposer').value,
    group_id: document.getElementById('v-group').value,
  };
  const r = await api('/api/vote/propose-model', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  if (r) { toast('Proposal created: '+r.proposal_id, 'ok'); loadPanel('voting'); }
}

async function castVote() {
  const body = {
    proposal_id: document.getElementById('cv-proposal').value,
    voter_id: document.getElementById('cv-voter').value,
    choice: document.getElementById('cv-choice').value,
  };
  const r = await api('/api/vote/cast', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  if (r) { toast('Vote cast (weight: '+r.weight.toFixed(2)+')', 'ok'); loadPanel('voting'); }
}

async function createGroup() {
  const body = {
    name: document.getElementById('g-name').value,
    owner_id: document.getElementById('g-owner').value,
    description: document.getElementById('g-desc').value,
    visibility: document.getElementById('g-vis').value,
    max_members: +document.getElementById('g-max').value,
    require_approval: document.getElementById('g-approval').value === 'true',
    passphrase: document.getElementById('g-pass').value,
  };
  const r = await api('/api/group/create', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  if (r) { toast('Group created: '+r.group_id, 'ok'); loadPanel('groups'); }
}

async function joinGroup() {
  const body = {
    group_id: document.getElementById('jg-id').value,
    user_id: document.getElementById('jg-user').value,
    invite_code: document.getElementById('jg-invite').value,
    cpu_cores: +document.getElementById('jg-cpu').value,
    gpu_count: +document.getElementById('jg-gpu').value,
    ram_gb: +document.getElementById('jg-ram').value,
  };
  const r = await api('/api/group/join', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  if (r) { toast(r.message, 'ok'); loadPanel('groups'); }
}

async function pledgeResources() {
  const body = {
    user_id: document.getElementById('p-user').value,
    node_id: document.getElementById('p-node').value,
    cpu_cores: +document.getElementById('p-cpu').value,
    gpu_count: +document.getElementById('p-gpu').value,
    ram_gb: +document.getElementById('p-ram').value,
    time_hours: +document.getElementById('p-time').value,
    group_id: document.getElementById('p-group').value,
  };
  const r = await api('/api/pledge', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  if (r) { toast('Pledged! Score: '+r.score.toFixed(1)+' - '+r.summary, 'ok'); loadPanel('pledges'); }
}

async function loadModel() {
  const body = {
    model_id: document.getElementById('inf-model').value,
    model_name: document.getElementById('inf-model').value,
    version: document.getElementById('inf-version').value,
    num_shards: +document.getElementById('inf-shards').value,
    device: document.getElementById('inf-device').value,
    mirror_enabled: document.getElementById('inf-mirror').value === 'true',
    group_id: document.getElementById('inf-group').value,
  };
  const r = await api('/api/inference/load', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  if (r) { toast('Model loaded: '+r.model_id+' (shards: '+r.shards+')', 'ok'); loadPanel('inference'); }
}

async function runInference() {
  const out = document.getElementById('inf-output');
  out.style.display = 'block';
  out.textContent = 'Generating...';
  const body = {
    model_id: document.getElementById('inf-run-model').value,
    prompt: document.getElementById('inf-prompt').value,
    max_tokens: +document.getElementById('inf-max-tokens').value,
    temperature: +document.getElementById('inf-temp').value,
  };
  const r = await api('/api/inference/run', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  if (r) {
    out.textContent = r.text || r.error || JSON.stringify(r, null, 2);
    if (r.tokens_per_second) out.textContent += '\\n\\nSpeed: ' + r.tokens_per_second.toFixed(1) + ' tok/s | Latency: ' + r.latency_ms.toFixed(0) + 'ms';
  }
}

async function jackIn() {
  const jiResult = document.getElementById('ji-result');
  jiResult.style.display = 'block';
  jiResult.innerHTML = '<em>Jacking in...</em>';
  const models = document.getElementById('ji-models').value.split(',').map(s=>s.trim()).filter(Boolean);
  const body = {
    user_id: document.getElementById('ji-user').value,
    mode: document.getElementById('ji-mode').value,
    cpu_cores: +document.getElementById('ji-cpu').value,
    gpu_count: +document.getElementById('ji-gpu').value,
    ram_gb: +document.getElementById('ji-ram').value,
    time_hours: +document.getElementById('ji-hours').value,
    group_id: document.getElementById('ji-group').value,
    models_to_serve: models,
  };
  const r = await api('/api/jack-in', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  if (r) {
    let html = '<h4 style="color:#00d2ff">Jacked In!</h4>';
    html += '<p>Modes: <b>' + r.modes.join(' + ') + '</b></p>';
    if (r.pledge_score) html += '<p>Training pledge score: <b>' + r.pledge_score.toFixed(1) + '</b> (' + r.pledge_summary + ')</p>';
    if (r.group_joined) html += '<p>Joined group: <b>' + r.group_joined + '</b></p>';
    if (r.profile) html += '<p>Profile: CPU ' + r.profile.cpu + ' | GPU ' + r.profile.gpu + ' | RAM ' + r.profile.ram + '</p>';
    jiResult.innerHTML = html;
    toast('Jacked in! Modes: ' + r.modes.join(', '), 'ok');
  }
}

loadPanel('overview');
refreshInterval = setInterval(() => {
  const active = document.querySelector('.panel.active');
  if (active) loadPanel(active.id);
}, 10000);

// Gradient Sync Functions
async function addGradientPeer() {
  const nodeId = document.getElementById('gs-peer-id').value;
  const endpoint = document.getElementById('gs-peer-ep').value;
  if (!nodeId || !endpoint) { toast('Enter peer ID and endpoint', 'err'); return; }
  const r = await api(`/api/training/gradient-peer?node_id=${encodeURIComponent(nodeId)}&endpoint=${encodeURIComponent(endpoint)}`, {method:'POST'});
  if (r) { toast('Peer added: ' + nodeId, 'ok'); loadPanel('gradients'); }
}

async function aggregateGradients() {
  const jobId = document.getElementById('gs-job').value;
  const step = document.getElementById('gs-step').value;
  if (!jobId) { toast('Enter Job ID', 'err'); return; }
  const r = await api(`/api/training/gradient-aggregate/${jobId}/${step}`, {method:'POST'});
  if (r) { toast('Aggregated: ' + (r.layers||0) + ' layers', 'ok'); }
}

async function pushGradients() {
  const jobId = document.getElementById('gs-job').value;
  const step = document.getElementById('gs-step').value;
  if (!jobId) { toast('Enter Job ID', 'err'); return; }
  const r = await api(`/api/training/gradient-push/${jobId}/${step}`, {method:'POST'});
  if (r) { toast('Push: ' + r.status, 'ok'); }
}

async function pullGradients() {
  const jobId = document.getElementById('gs-job').value;
  const step = document.getElementById('gs-step').value;
  if (!jobId) { toast('Enter Job ID', 'err'); return; }
  const r = await api(`/api/training/gradient-pull/${jobId}/${step}`);
  if (r && r.gradients) {
    let html = '<table><tr><th>Layer</th><th>Shape</th><th>Mean</th><th>Std</th><th>Norm</th></tr>';
    for (const [layer, g] of Object.entries(r.gradients)) {
      html += `<tr><td>${layer}</td><td>${g.shape.join('x')}</td><td>${g.mean.toFixed(4)}</td><td>${g.std.toFixed(4)}</td><td>${g.norm.toFixed(2)}</td></tr>`;
    }
    html += '</table>';
    document.getElementById('gs-status').innerHTML = html;
  } else {
    toast('No gradients found', 'err');
  }
}

// Metrics Functions
async function loadMetrics() {
  const r = await fetch('/api/metrics');
  const text = await r.text();
  document.getElementById('metrics-content').textContent = text;
}

// Auth Functions
async function createAuthToken() {
  const userId = document.getElementById('at-user').value;
  const ttl = parseFloat(document.getElementById('at-ttl').value);
  const r = await api('/api/auth/token', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({user_id: userId, scopes: ['read','write'], ttl_hours: ttl})});
  if (r) {
    const el = document.getElementById('at-result');
    el.style.display = 'block';
    el.innerHTML = '<div class="card"><p>Token: <code style="word-break:break-all;color:#00ff88">' + r.token + '</code></p><p>Expires: ' + r.expires_in_hours + ' hours | Scopes: ' + r.scopes.join(', ') + '</p></div>';
    toast('Token created!', 'ok');
  }
}

async function createApiKey() {
  const userId = document.getElementById('ak-user').value;
  const name = document.getElementById('ak-name').value;
  const r = await api('/api/auth/api-key', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({user_id: userId, name: name, scopes: ['read','write']})});
  if (r) {
    const el = document.getElementById('ak-result');
    el.style.display = 'block';
    el.innerHTML = '<div class="card"><p>API Key: <code style="word-break:break-all;color:#00ff88">' + r.api_key + '</code></p><p>Name: ' + r.name + ' | Scopes: ' + r.scopes.join(', ') + '</p></div>';
    toast('API key created!', 'ok');
  }
}

async function verifyToken() {
  const token = document.getElementById('vt-token').value;
  if (!token) { toast('Enter a token', 'err'); return; }
  const r = await api(`/api/auth/verify?token=${encodeURIComponent(token)}`);
  const el = document.getElementById('vt-result');
  el.style.display = 'block';
  if (r && r.valid) {
    el.innerHTML = '<div class="card" style="border-color:#00cc66"><p style="color:#00cc66">Valid Token</p><p>User: ' + r.user_id + '</p><p>Scopes: ' + r.scopes.join(', ') + '</p></div>';
  } else {
    el.innerHTML = '<div class="card" style="border-color:#cc3333"><p style="color:#cc3333">Invalid or expired token</p></div>';
  }
}
</script>
</body>
</html>"""