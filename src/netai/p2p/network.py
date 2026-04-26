"""P2P networking layer - discovery, heartbeats, peer exchange."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import aiohttp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NodeState(str, Enum):
    JOINING = "joining"
    ACTIVE = "active"
    TRAINING = "training"
    DRAINING = "draining"
    OFFLINE = "offline"


class PeerInfo(BaseModel):
    node_id: str
    host: str
    port: int
    state: NodeState = NodeState.JOINING
    cpu_cores: int = 0
    cpu_avail: int = 0
    gpu_count: int = 0
    gpu_avail: int = 0
    gpu_names: list[str] = Field(default_factory=list)
    ram_gb: float = 0.0
    ram_avail_gb: float = 0.0
    last_heartbeat: float = 0.0
    active_jobs: int = 0
    version: str = "1.0.0"

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_alive(self) -> bool:
        return (
            self.state not in (NodeState.OFFLINE,)
            and (time.time() - self.last_heartbeat) < 90
        )


class PeerMessage(BaseModel):
    msg_type: str
    sender_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    signature: str = ""


@dataclass
class PeerRecord:
    info: PeerInfo
    added_at: float = field(default_factory=time.time)
    reliability_score: float = 1.0


class PeerTable:
    def __init__(self, self_id: str, max_peers: int = 256):
        self.self_id = self_id
        self.max_peers = max_peers
        self._peers: dict[str, PeerRecord] = {}
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def add_peer(self, info: PeerInfo) -> bool:
        if info.node_id == self.self_id:
            return False
        async with self._get_lock():
            if info.node_id in self._peers:
                existing = self._peers[info.node_id]
                existing.info = info
                existing.info.last_heartbeat = time.time()
                return True
            if len(self._peers) >= self.max_peers:
                worst = min(self._peers.items(), key=lambda x: x[1].reliability_score)
                del self._peers[worst[0]]
            self._peers[info.node_id] = PeerRecord(info=info)
            logger.info("Peer added: %s (%s:%d)", info.node_id, info.host, info.port)
            return True

    async def remove_peer(self, node_id: str) -> bool:
        async with self._get_lock():
            return self._peers.pop(node_id, None) is not None

    async def get_alive_peers(self) -> list[PeerInfo]:
        async with self._get_lock():
            now = time.time()
            alive = []
            for nid, rec in list(self._peers.items()):
                if rec.info.is_alive:
                    alive.append(rec.info)
                elif now - rec.info.last_heartbeat > 300:
                    rec.reliability_score *= 0.5
                    if rec.reliability_score < 0.1:
                        del self._peers[nid]
            return alive

    async def get_all_peers(self) -> list[PeerInfo]:
        async with self._get_lock():
            return [r.info for r in self._peers.values()]

    async def update_reliability(self, node_id: str, success: bool):
        async with self._get_lock():
            if node_id in self._peers:
                delta = 0.05 if success else -0.15
                rec = self._peers[node_id]
                rec.reliability_score = max(0.0, min(2.0, rec.reliability_score + delta))


class P2PNode:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 7999,
        seed_nodes: list[str] | None = None,
        node_id: str | None = None,
        node_identity: "NodeIdentity | None" = None,
    ):
        self.host = host
        self.port = port
        self.node_id = node_id or self._generate_node_id()
        self.seed_nodes = seed_nodes or []
        self.peer_table = PeerTable(self.node_id)
        self.state = NodeState.JOINING
        self._server: aiohttp.web.Application | None = None
        self._runner: aiohttp.web.AppRunner | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._discovery_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._on_peer_update_callbacks: list[Any] = []
        self._session: aiohttp.ClientSession | None = None
        self._handlers: dict[str, Any] = {}
        self._node_identity = node_identity
        self._known_public_keys: dict[str, bytes] = {}
        self._cached_profile: Any = None
        self._cached_profile_time: float = 0.0

    def _sign_message(self, msg: PeerMessage) -> dict:
        msg_dict = msg.model_dump()
        if self._node_identity:
            clean_dict = {k: v for k, v in msg_dict.items() if k not in ("signature", "signature_algo")}
            msg_bytes = json.dumps(clean_dict, sort_keys=True, separators=(",", ":")).encode()
            sig = self._node_identity.sign(msg_bytes)
            msg_dict["signature"] = sig.hex()
            msg_dict["signature_algo"] = "ed25519"
        return msg_dict

    @staticmethod
    def _generate_node_id() -> str:
        return secrets.token_hex(16)

    def on(self, msg_type: str, handler):
        self._handlers[msg_type] = handler

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        return self._session

    def _create_app(self) -> aiohttp.web.Application:
        app = aiohttp.web.Application()
        app.router.add_post("/p2p/join", self._handle_join)
        app.router.add_post("/p2p/heartbeat", self._handle_heartbeat)
        app.router.add_get("/p2p/peers", self._handle_get_peers)
        app.router.add_post("/p2p/message", self._handle_message)
        app.router.add_get("/p2p/status", self._handle_status)
        app.router.add_post("/p2p/leave", self._handle_leave)
        return app

    async def start(self):
        self._server = self._create_app()
        self._runner = aiohttp.web.AppRunner(self._server)
        await self._runner.setup()
        site = aiohttp.web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        self.state = NodeState.ACTIVE
        logger.info("P2P node %s started on %s:%d", self.node_id, self.host, self.port)
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        await self._bootstrap()

    async def stop(self):
        await self._announce_leave()
        for t in [self._heartbeat_task, self._discovery_task, self._cleanup_task]:
            if t and not t.done():
                t.cancel()
        if self._runner:
            await self._runner.cleanup()
        if self._session and not self._session.closed:
            await self._session.close()
        self.state = NodeState.OFFLINE
        logger.info("P2P node %s stopped", self.node_id)

    async def _bootstrap(self, max_retries: int = 5, base_delay: float = 2.0):
        for attempt in range(max_retries):
            for seed in self.seed_nodes:
                try:
                    if ":" not in seed:
                        logger.warning("Invalid seed node (missing port): %s", seed)
                        continue
                    host, port_str = seed.rsplit(":", 1)
                    port = int(port_str)
                    if not (1 <= port <= 65535):
                        logger.warning("Invalid seed port %d: %s", port, seed)
                        continue
                    if await self._join_peer(host, port):
                        return
                except Exception as e:
                    logger.warning("Bootstrap attempt %d failed for %s: %s", attempt + 1, seed, e)
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info("Bootstrap retry %d/%d in %.1fs", attempt + 1, max_retries, delay)
                await asyncio.sleep(delay)

    async def _join_peer(self, host: str, port: int) -> bool:
        session = await self._get_session()
        self_info = await self._get_self_info()
        try:
            async with session.post(
                f"http://{host}:{port}/p2p/join",
                json=self_info.model_dump(),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    remote_node_id = data.get("node_id", "")
                    for p in data.get("peers", []):
                        pi = PeerInfo(**p)
                        if pi.node_id != self.node_id:
                            await self.peer_table.add_peer(pi)
                    if remote_node_id:
                        await self.peer_table.update_reliability(remote_node_id, True)
                    logger.info("Joined peer %s:%d", host, port)
                    return True
        except Exception as e:
            logger.debug("Join failed %s:%d: %s", host, port, e)
        return False

    async def _announce_leave(self):
        alive = await self.peer_table.get_alive_peers()
        session = await self._get_session()
        for peer in alive:
            try:
                async with session.post(
                    f"{peer.endpoint}/p2p/leave",
                    json={"node_id": self.node_id},
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    pass
            except Exception as e:
                logger.debug("Seed node join failed: %s", e)

    async def _get_self_info(self) -> PeerInfo:
        if self._cached_profile is None or (time.time() - self._cached_profile_time > 300):
            from netai.resource.profiler import ResourceProfiler
            prof = ResourceProfiler()
            res = prof.profile()
            self._cached_profile = res
            self._cached_profile_time = time.time()
        else:
            res = self._cached_profile
        return PeerInfo(
            node_id=self.node_id,
            host=self.host if self.host != "0.0.0.0" else self._get_local_ip(),
            port=self.port,
            state=self.state,
            cpu_cores=res.cpu_cores,
            cpu_avail=res.cpu_available,
            gpu_count=res.gpu_count,
            gpu_avail=res.gpu_available,
            gpu_names=res.gpu_names,
            ram_gb=res.ram_total_gb,
            ram_avail_gb=res.ram_available_gb,
            last_heartbeat=time.time(),
            version="1.0.0",
        )

    @staticmethod
    def _get_local_ip() -> str:
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            return ip
        except Exception:
            return "127.0.0.1"
        finally:
            if s:
                s.close()

    async def _heartbeat_loop(self):
        while True:
            try:
                await asyncio.sleep(30)
                self_info = await self._get_self_info()
                alive = await self.peer_table.get_alive_peers()
                session = await self._get_session()

                async def _send_heartbeat(peer):
                    try:
                        async with session.post(
                            f"{peer.endpoint}/p2p/heartbeat",
                            json=self_info.model_dump(),
                            timeout=aiohttp.ClientTimeout(total=5),
                        ) as resp:
                            if resp.status != 200:
                                await self.peer_table.update_reliability(peer.node_id, False)
                    except Exception:
                        await self.peer_table.update_reliability(peer.node_id, False)

                await asyncio.gather(*[_send_heartbeat(p) for p in alive[:20]], return_exceptions=True)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error: %s", e)

    async def _discovery_loop(self):
        while True:
            try:
                await asyncio.sleep(60)
                alive = await self.peer_table.get_alive_peers()
                session = await self._get_session()
                for peer in alive[:5]:
                    try:
                        async with session.get(
                            f"{peer.endpoint}/p2p/peers",
                            timeout=aiohttp.ClientTimeout(total=5),
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                for p in data.get("peers", []):
                                    pi = PeerInfo(**p)
                                    if pi.node_id != self.node_id and pi.is_alive:
                                        await self.peer_table.add_peer(pi)
                    except Exception:
                        pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Discovery error: %s", e)

    async def _cleanup_loop(self):
        while True:
            try:
                await asyncio.sleep(120)
                now = time.time()
                all_peers = await self.peer_table.get_all_peers()
                for p in all_peers:
                    if now - p.last_heartbeat > 300:
                        await self.peer_table.remove_peer(p.node_id)
                        logger.info("Removed stale peer %s", p.node_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup error: %s", e)

    async def broadcast(self, msg_type: str, payload: dict[str, Any]):
        alive = await self.peer_table.get_alive_peers()
        msg = PeerMessage(
            msg_type=msg_type,
            sender_id=self.node_id,
            payload=payload,
        )
        if self._node_identity:
            msg_dict = self._sign_message(msg)
        else:
            msg_dict = msg.model_dump()
        session = await self._get_session()

        async def _send_to_peer(peer):
            try:
                async with session.post(
                    f"{peer.endpoint}/p2p/message",
                    json=msg_dict,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    pass
            except Exception as e:
                logger.debug("Broadcast to peer %s failed: %s", peer.node_id, e)

        await asyncio.gather(*[_send_to_peer(p) for p in alive], return_exceptions=True)

    async def send_to(self, node_id: str, msg_type: str, payload: dict[str, Any]) -> dict | None:
        all_peers = await self.peer_table.get_all_peers()
        target = None
        for p in all_peers:
            if p.node_id == node_id:
                target = p
                break
        if not target:
            return None
        msg = PeerMessage(msg_type=msg_type, sender_id=self.node_id, payload=payload)
        if self._node_identity:
            msg_dict = self._sign_message(msg)
        else:
            msg_dict = msg.model_dump()
        session = await self._get_session()
        try:
            async with session.post(
                f"{target.endpoint}/p2p/message",
                json=msg_dict,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.debug("Health check for %s failed: %s", node_id, e)
        return None

    async def _handle_join(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        try:
            data = await request.json()
        except Exception:
            return aiohttp.web.json_response({"status": "error", "message": "Invalid JSON"}, status=400)
        if len(str(data)) > 65536:
            return aiohttp.web.json_response({"status": "error", "message": "Payload too large"}, status=413)
        peer = PeerInfo(**data)
        peer.last_heartbeat = time.time()
        host = peer.host or ""
        if host in ("127.0.0.1", "::1", "localhost", "0.0.0.0") and peer.node_id != self.node_id:
            return aiohttp.web.json_response({"status": "error", "message": "Invalid host address"}, status=400)
        if peer.cpu_cores < 0 or peer.gpu_count < 0 or peer.ram_gb < 0:
            return aiohttp.web.json_response({"status": "error", "message": "Invalid resource values"}, status=400)
        existing = self.peer_table.peers.get(peer.node_id)
        if existing and existing.host != peer.host:
            existing.host = peer.host
        await self.peer_table.add_peer(peer)
        peers = await self.peer_table.get_alive_peers()
        peer_list = [p.model_dump() for p in peers if p.node_id != peer.node_id]
        return aiohttp.web.json_response(
            {"status": "ok", "node_id": self.node_id, "peers": peer_list}
        )

    async def _handle_heartbeat(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        try:
            data = await request.json()
        except Exception:
            return aiohttp.web.json_response({"status": "error", "message": "Invalid JSON"}, status=400)
        peer = PeerInfo(**data)
        peer.last_heartbeat = time.time()
        await self.peer_table.add_peer(peer)
        self_info = await self._get_self_info()
        return aiohttp.web.json_response({"status": "ok", "state": self_info.state.value})

    async def _handle_get_peers(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        alive = await self.peer_table.get_alive_peers()
        peer_list = [p.model_dump() for p in alive]
        return aiohttp.web.json_response({"peers": peer_list, "count": len(alive)})

    async def _handle_message(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        content_length = request.content_length or 0
        if content_length > 5 * 1024 * 1024:
            return aiohttp.web.json_response({"status": "error", "message": "Message too large (max 5MB)"}, status=413)
        try:
            data = await request.json()
        except Exception:
            return aiohttp.web.json_response({"status": "error", "message": "Invalid JSON"}, status=400)
        sig_hex = data.get("signature", "")
        sender_id = data.get("sender_id", "")
        if self._node_identity and sender_id:
            if not sig_hex:
                return aiohttp.web.json_response({"status": "error", "message": "signature required"}, status=403)
            if sender_id not in self._known_public_keys:
                pk_hex = data.get("sender_public_key", "")
                if not pk_hex:
                    logger.warning("P2P message from unknown sender %s (no public key)", sender_id)
                    return aiohttp.web.json_response({"status": "error", "message": "unknown sender"}, status=403)
                try:
                    self._known_public_keys[sender_id] = bytes.fromhex(pk_hex)
                except ValueError:
                    logger.warning("P2P message from sender %s has invalid public key", sender_id)
                    return aiohttp.web.json_response({"status": "error", "message": "invalid public key"}, status=403)
            try:
                from cryptography.hazmat.primitives.asymmetric import ed25519 as ed
                clean_data = {k: v for k, v in data.items() if k not in ("signature", "signature_algo")}
                msg_bytes = json.dumps(clean_data, sort_keys=True, separators=(",", ":")).encode()
                vk = ed.Ed25519PublicKey.from_public_bytes(self._known_public_keys[sender_id])
                vk.verify(bytes.fromhex(sig_hex), msg_bytes)
            except Exception:
                logger.warning("P2P message signature verification failed from %s", sender_id)
                return aiohttp.web.json_response({"status": "signature_invalid"}, status=403)
        msg = PeerMessage(
            msg_type=data.get("msg_type", ""),
            sender_id=sender_id,
            payload=data.get("payload", {}),
            signature=sig_hex,
        )
        handler = self._handlers.get(msg.msg_type)
        if handler:
            result = await handler(msg)
            return aiohttp.web.json_response(result or {"status": "ok"})
        return aiohttp.web.json_response({"status": "no_handler", "msg_type": msg.msg_type})

    async def _handle_status(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        self_info = await self._get_self_info()
        alive = await self.peer_table.get_alive_peers()
        return aiohttp.web.json_response({
            "node_id": self.node_id,
            "state": self.state.value,
            "self": self_info.model_dump(),
            "peer_count": len(alive),
        })

    async def _handle_leave(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        data = await request.json()
        node_id = data.get("node_id", "")
        remote_ip = request.remote or "unknown"
        if not node_id:
            return aiohttp.web.json_response({"status": "error", "message": "node_id required"}, status=400)
        sender_matches = node_id == self.node_id
        if not sender_matches and remote_ip in ("127.0.0.1", "::1", "localhost"):
            if request.headers.get("X-Node-Id") == self.node_id or data.get("signature"):
                sender_matches = True
        if sender_matches:
            await self.peer_table.remove_peer(node_id)
            return aiohttp.web.json_response({"status": "ok", "removed": node_id})
        return aiohttp.web.json_response({"status": "error", "message": "not authorized to remove peer"}, status=403)


async def create_p2p_node(
    host: str = "0.0.0.0",
    port: int = 7999,
    seed_nodes: list[str] | None = None,
    node_id: str | None = None,
) -> P2PNode:
    node = P2PNode(host=host, port=port, seed_nodes=seed_nodes, node_id=node_id)
    await node.start()
    return node