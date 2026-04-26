"""Federation protocol - connect multiple NetAI clusters together."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from typing import Any

import aiohttp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FederationNode(BaseModel):
    node_id: str
    endpoint: str
    name: str = ""
    cluster_id: str = ""
    cpu_cores: int = 0
    gpu_count: int = 0
    ram_gb: float = 0.0
    peer_count: int = 0
    active_jobs: int = 0
    last_sync: float = 0.0
    trust_score: float = 1.0
    region: str = ""


class FederationProposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    proposer_cluster: str = ""
    title: str = ""
    description: str = ""
    resource_request: dict[str, Any] = Field(default_factory=dict)
    resource_offer: dict[str, Any] = Field(default_factory=dict)
    training_config: dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"
    votes: dict[str, str] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    deadline: float = 0.0


class FederationSyncMessage(BaseModel):
    msg_type: str
    sender_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    signature: str = ""


class Federation:
    def __init__(self, cluster_id: str, cluster_endpoint: str = "http://localhost:8001"):
        self.cluster_id = cluster_id
        self.cluster_endpoint = cluster_endpoint
        self.peers: dict[str, FederationNode] = {}
        self.proposals: dict[str, FederationProposal] = {}
        self._session: aiohttp.ClientSession | None = None
        self._sync_task: asyncio.Task | None = None
        self._running = False
        self.trust_threshold = 0.5
        self._handlers: dict[str, Any] = {}

    def on(self, msg_type: str, handler):
        self._handlers[msg_type] = handler

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        return self._session

    async def start(self):
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info("Federation %s started", self.cluster_id)

    async def stop(self):
        self._running = False
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
        if self._session and not self._session.closed:
            await self._session.close()

    async def register_peer(self, node: FederationNode) -> bool:
        if node.node_id == self.cluster_id:
            return False
        node.last_sync = time.time()
        self.peers[node.node_id] = node
        logger.info("Federation peer registered: %s (%s)", node.name, node.node_id)
        return True

    async def unregister_peer(self, node_id: str):
        self.peers.pop(node_id, None)

    async def propose_shared_training(
        self,
        title: str,
        description: str,
        resource_request: dict[str, Any],
        training_config: dict[str, Any],
        deadline_hours: float = 168.0,
    ) -> FederationProposal:
        proposal = FederationProposal(
            proposer_cluster=self.cluster_id,
            title=title,
            description=description,
            resource_request=resource_request,
            training_config=training_config,
            deadline=time.time() + deadline_hours * 3600,
        )
        self.proposals[proposal.proposal_id] = proposal
        await self._broadcast(FederationSyncMessage(
            msg_type="federation_proposal",
            sender_id=self.cluster_id,
            payload=proposal.model_dump(),
        ))
        return proposal

    async def vote_on_proposal(self, proposal_id: str, vote: str) -> bool:
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return False
        if proposal.status != "pending":
            return False
        if time.time() > proposal.deadline:
            proposal.status = "expired"
            return False
        proposal.votes[self.cluster_id] = vote
        for_vote = sum(1 for v in proposal.votes.values() if v == "approve")
        total = len(proposal.votes)
        if total > 0:
            approve_pct = for_vote / total
            if approve_pct >= 0.6:
                proposal.status = "approved"
            elif (total - for_vote) / total > 0.6:
                proposal.status = "rejected"
        await self._broadcast(FederationSyncMessage(
            msg_type="federation_vote",
            sender_id=self.cluster_id,
            payload={"proposal_id": proposal_id, "vote": vote},
        ))
        return True

    async def offer_resources(self, target_cluster: str, resources: dict[str, Any]):
        msg = FederationSyncMessage(
            msg_type="resource_offer",
            sender_id=self.cluster_id,
            payload={
                "target": target_cluster,
                "resources": resources,
                "timestamp": time.time(),
            },
        )
        await self._send_to(target_cluster, msg)

    async def _sync_loop(self):
        while self._running:
            try:
                await asyncio.sleep(60)
                for peer_id, peer in list(self.peers.items()):
                    try:
                        session = await self._get_session()
                        async with session.get(
                            f"{peer.endpoint}/api/status",
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                peer.active_jobs = len(data.get("jobs", []))
                                peer.last_sync = time.time()
                                peer.peer_count = data.get("peer_count", 0)
                                peer.trust_score = min(1.0, peer.trust_score + 0.02)
                    except Exception:
                        peer.trust_score = max(0.1, peer.trust_score * 0.95)
                stale = [pid for pid, p in self.peers.items() if time.time() - p.last_sync > 600]
                for pid in stale:
                    self.peers.pop(pid, None)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Federation sync error: %s", e)

    async def _broadcast(self, msg: FederationSyncMessage):
        session = await self._get_session()
        for peer in self.peers.values():
            if peer.trust_score >= self.trust_threshold:
                try:
                    async with session.post(
                        f"{peer.endpoint}/api/federation/message",
                        json=msg.model_dump(),
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        pass
                except Exception:
                    peer.trust_score *= 0.9

    async def _send_to(self, cluster_id: str, msg: FederationSyncMessage):
        peer = self.peers.get(cluster_id)
        if not peer:
            return
        session = await self._get_session()
        try:
            async with session.post(
                f"{peer.endpoint}/api/federation/message",
                json=msg.model_dump(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                return await resp.json() if resp.status == 200 else None
        except Exception:
            peer.trust_score *= 0.9
        return None

    def get_status(self) -> dict[str, Any]:
        total_cpu = sum(p.cpu_cores for p in self.peers.values())
        total_gpu = sum(p.gpu_count for p in self.peers.values())
        total_ram = sum(p.ram_gb for p in self.peers.values())
        return {
            "cluster_id": self.cluster_id,
            "cluster_endpoint": self.cluster_endpoint,
            "trust_threshold": self.trust_threshold,
            "federated_peers": len(self.peers),
            "total_available_cpu": total_cpu,
            "total_available_gpu": total_gpu,
            "total_available_ram_gb": total_ram,
            "active_proposals": sum(1 for p in self.proposals.values() if p.status == "pending"),
            "approved_proposals": sum(1 for p in self.proposals.values() if p.status == "approved"),
            "peers": [
                {
                    "node_id": p.node_id,
                    "name": p.name,
                    "endpoint": p.endpoint,
                    "trust_score": p.trust_score,
                    "last_sync_ago": time.time() - p.last_sync if p.last_sync > 0 else -1,
                }
                for p in self.peers.values()
            ],
        }

    def get_aggregate_resources(self) -> dict[str, Any]:
        local_resources = {}
        for peer in self.peers.values():
            if peer.trust_score >= self.trust_threshold:
                local_resources[peer.node_id] = {
                    "cpu_cores": peer.cpu_cores,
                    "gpu_count": peer.gpu_count,
                    "ram_gb": peer.ram_gb,
                    "trust_score": peer.trust_score,
                    "active_jobs": peer.active_jobs,
                }
        return {
            "peers": local_resources,
            "total_cpu": sum(p["cpu_cores"] for p in local_resources.values()),
            "total_gpu": sum(p["gpu_count"] for p in local_resources.values()),
            "total_ram": sum(p["ram_gb"] for p in local_resources.values()),
        }