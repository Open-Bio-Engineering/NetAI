"""Job scheduler - assigns training jobs to available nodes based on resources."""

from __future__ import annotations

import heapq
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class JobPriority(int, Enum):
    URGENT = 1
    HIGH = 5
    NORMAL = 10
    LOW = 20


class SchedulePolicy(str, Enum):
    FIFO = "fifo"
    PRIORITY = "priority"
    FAIR_SHARE = "fair_share"
    RESOURCE_AWARE = "resource_aware"


class NodeResources(BaseModel):
    node_id: str
    cpu_cores: int = 0
    cpu_available: int = 0
    gpu_count: int = 0
    gpu_available: int = 0
    ram_gb: float = 0.0
    ram_available_gb: float = 0.0
    gpu_vram_mb: list[int] = Field(default_factory=list)
    gpu_available_vram_mb: list[int] = Field(default_factory=list)
    is_training: bool = False
    current_jobs: int = 0
    max_jobs: int = 3
    reliability: float = 1.0
    group_id: str = ""
    last_seen: float = 0.0

    @property
    def capacity_score(self) -> float:
        score = self.cpu_available * 1.0
        score += self.gpu_available * 10.0
        score += min(self.ram_available_gb, 128.0) * 0.5
        for vram in self.gpu_available_vram_mb:
            score += vram / 1024.0 * 2.0
        return score

    @property
    def is_available(self) -> bool:
        return (
            self.current_jobs < self.max_jobs
            and (self.cpu_available > 0 or self.gpu_available > 0)
        )


class JobRequirements(BaseModel):
    min_cpu_cores: int = 1
    min_gpu_count: int = 0
    min_ram_gb: float = 4.0
    min_gpu_vram_mb: int = 0
    preferred_device: str = "cuda"
    max_duration_hours: float = 24.0
    group_id: str = ""
    priority: int = JobPriority.NORMAL


class ScheduledJob(BaseModel):
    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    requirements: JobRequirements = Field(default_factory=JobRequirements)
    assigned_nodes: list[str] = Field(default_factory=list)
    status: str = "pending"
    created_at: float = Field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    submitter_id: str = ""
    group_id: str = ""


@dataclass(order=True)
class PriorityJob:
    priority: int
    created_at: float
    job_id: str = field(compare=False)
    requirements: JobRequirements = field(compare=False, default_factory=JobRequirements)
    name: str = field(compare=False, default="")
    submitter_id: str = field(compare=False, default="")
    group_id: str = field(compare=False, default="")


class JobScheduler:
    def __init__(self, policy: SchedulePolicy = SchedulePolicy.RESOURCE_AWARE):
        self.policy = policy
        self.nodes: dict[str, NodeResources] = {}
        self.queue: list[PriorityJob] = []
        self.running_jobs: dict[str, ScheduledJob] = {}
        self.completed_jobs: dict[str, ScheduledJob] = {}
        self._fair_share_counters: dict[str, int] = {}
        self._assignment_history: list[dict] = []

    def register_node(self, resources: NodeResources):
        self.nodes[resources.node_id] = resources
        logger.info("Node registered: %s (CPU:%d GPU:%d RAM:%.1fGB)",
                     resources.node_id, resources.cpu_available, resources.gpu_available,
                     resources.ram_available_gb)

    def unregister_node(self, node_id: str):
        self.nodes.pop(node_id, None)
        for job in list(self.running_jobs.values()):
            if node_id in job.assigned_nodes:
                job.assigned_nodes.remove(node_id)
                if not job.assigned_nodes:
                    job.status = "interrupted"

    def submit_job(self, requirements: JobRequirements, name: str = "", submitter_id: str = "") -> str:
        job_id = uuid.uuid4().hex[:12]
        pj = PriorityJob(
            priority=requirements.priority,
            created_at=time.time(),
            job_id=job_id,
            requirements=requirements,
            name=name,
            submitter_id=submitter_id,
            group_id=requirements.group_id,
        )
        heapq.heappush(self.queue, pj)
        logger.info("Job submitted: %s (%s) priority=%d", job_id, name, requirements.priority)
        return job_id

    def schedule(self) -> list[tuple[str, list[str]]]:
        assignments = []
        remaining = []
        while self.queue:
            pj = heapq.heappop(self.queue)
            nodes = self._find_best_nodes(pj.requirements)
            if nodes:
                sj = ScheduledJob(
                    job_id=pj.job_id,
                    name=pj.name,
                    requirements=pj.requirements,
                    assigned_nodes=[n.node_id for n in nodes],
                    status="running",
                    started_at=time.time(),
                    submitter_id=pj.submitter_id,
                    group_id=pj.group_id,
                )
                for node in nodes:
                    node.current_jobs += 1
                    node.is_training = True
                self.running_jobs[pj.job_id] = sj
                assignments.append((pj.job_id, sj.assigned_nodes))
                self._assignment_history.append({
                    "job_id": pj.job_id,
                    "nodes": sj.assigned_nodes,
                    "timestamp": time.time(),
                })
                self._fair_share_counters[pj.submitter_id] = self._fair_share_counters.get(pj.submitter_id, 0) + 1
            else:
                remaining.append(pj)
        for pj in remaining:
            heapq.heappush(self.queue, pj)
        return assignments

    def _find_best_nodes(self, req: JobRequirements) -> list[NodeResources]:
        candidates = [
            n for n in self.nodes.values()
            if n.is_available
            and (not req.group_id or n.group_id == req.group_id)
        ]
        if req.min_gpu_count > 0:
            gpu_nodes = [n for n in candidates if n.gpu_available >= req.min_gpu_count]
            if not gpu_nodes:
                return []
            scored = sorted(gpu_nodes, key=lambda n: (
                -n.gpu_available,
                -n.capacity_score,
                -n.reliability,
            ))
            best = scored[0]
            if best.ram_available_gb < req.min_ram_gb:
                return []
            if req.min_gpu_vram_mb > 0:
                best_vram = max(best.gpu_available_vram_mb) if best.gpu_available_vram_mb else 0
                if best_vram < req.min_gpu_vram_mb:
                    return []
            return [best]
        else:
            scored = sorted(candidates, key=lambda n: (
                -n.capacity_score,
                -n.reliability,
                n.current_jobs,
            ))
            for node in scored:
                if node.cpu_available >= req.min_cpu_cores and node.ram_available_gb >= req.min_ram_gb:
                    return [node]
            return []

    def complete_job(self, job_id: str, success: bool = True):
        job = self.running_jobs.pop(job_id, None)
        if not job:
            return
        for nid in job.assigned_nodes:
            node = self.nodes.get(nid)
            if node:
                node.current_jobs = max(0, node.current_jobs - 1)
                node.is_training = node.current_jobs > 0
        job.status = "completed" if success else "failed"
        job.completed_at = time.time()
        self.completed_jobs[job_id] = job

    def get_queue_status(self) -> dict[str, Any]:
        return {
            "queued": len(self.queue),
            "running": len(self.running_jobs),
            "completed": len(self.completed_jobs),
            "nodes_registered": len(self.nodes),
            "nodes_available": sum(1 for n in self.nodes.values() if n.is_available),
            "total_cpu": sum(n.cpu_cores for n in self.nodes.values()),
            "total_gpu": sum(n.gpu_count for n in self.nodes.values()),
            "total_ram_gb": sum(n.ram_gb for n in self.nodes.values()),
            "policy": self.policy.value,
        }

    def rebalance(self) -> list[tuple[str, str]]:
        migrations = []
        for job_id, job in list(self.running_jobs.items()):
            for nid in job.assigned_nodes:
                node = self.nodes.get(nid)
                if not node or not node.is_available:
                    new_nodes = self._find_best_nodes(job.requirements)
                    if new_nodes:
                        new_node = new_nodes[0]
                        old_node = self.nodes.get(nid)
                        if old_node:
                            old_node.current_jobs = max(0, old_node.current_jobs - 1)
                            old_node.is_training = old_node.current_jobs > 0
                        new_node.current_jobs += 1
                        new_node.is_training = True
                        job.assigned_nodes = [n if n != nid else new_node.node_id for n in job.assigned_nodes]
                        migrations.append((job_id, new_node.node_id))
        return migrations