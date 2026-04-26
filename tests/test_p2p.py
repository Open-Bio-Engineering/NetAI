"""Tests for P2P networking layer."""

import asyncio
import pytest
from netai.p2p.network import P2PNode, PeerInfo, PeerTable, NodeState, PeerMessage


class TestPeerInfo:
    def test_endpoint(self):
        p = PeerInfo(node_id="abc", host="192.168.1.1", port=7999)
        assert p.endpoint == "http://192.168.1.1:7999"

    def test_is_alive_active(self):
        import time
        p = PeerInfo(node_id="abc", host="h", port=1, state=NodeState.ACTIVE, last_heartbeat=time.time())
        assert p.is_alive is True

    def test_is_alive_offline(self):
        p = PeerInfo(node_id="abc", host="h", port=1, state=NodeState.OFFLINE)
        assert p.is_alive is False

    def test_is_alive_stale(self):
        p = PeerInfo(node_id="abc", host="h", port=1, state=NodeState.ACTIVE, last_heartbeat=0)
        assert p.is_alive is False


class TestPeerTable:
    @pytest.fixture
    def table(self):
        return PeerTable("self-node")

    @pytest.mark.asyncio
    async def test_add_peer(self, table):
        p = PeerInfo(node_id="peer1", host="h", port=1)
        result = await table.add_peer(p)
        assert result is True

    @pytest.mark.asyncio
    async def test_add_self(self, table):
        p = PeerInfo(node_id="self-node", host="h", port=1)
        result = await table.add_peer(p)
        assert result is False

    @pytest.mark.asyncio
    async def test_remove_peer(self, table):
        p = PeerInfo(node_id="peer1", host="h", port=1)
        await table.add_peer(p)
        result = await table.remove_peer("peer1")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_alive_peers(self, table):
        import time
        p1 = PeerInfo(node_id="p1", host="h", port=1, state=NodeState.ACTIVE, last_heartbeat=time.time())
        p2 = PeerInfo(node_id="p2", host="h", port=2, state=NodeState.OFFLINE)
        await table.add_peer(p1)
        await table.add_peer(p2)
        alive = await table.get_alive_peers()
        assert len(alive) == 1
        assert alive[0].node_id == "p1"

    @pytest.mark.asyncio
    async def test_update_reliability(self, table):
        p = PeerInfo(node_id="p1", host="h", port=1)
        await table.add_peer(p)
        await table.update_reliability("p1", True)
        peers = await table.get_all_peers()
        record = table._peers.get("p1")
        assert record.reliability_score == 1.05

    @pytest.mark.asyncio
    async def test_max_peers(self, table):
        table.max_peers = 3
        for i in range(5):
            p = PeerInfo(node_id=f"p{i}", host="h", port=i)
            await table.add_peer(p)
        all_p = await table.get_all_peers()
        assert len(all_p) <= 3


class TestPeerMessage:
    def test_create(self):
        msg = PeerMessage(msg_type="gradient", sender_id="node1", payload={"step": 1})
        assert msg.msg_type == "gradient"
        assert msg.sender_id == "node1"
        assert msg.payload["step"] == 1


class TestP2PNode:
    def test_generate_node_id(self):
        id1 = P2PNode._generate_node_id()
        id2 = P2PNode._generate_node_id()
        assert len(id1) == 32
        assert id1 != id2

    def test_init(self):
        node = P2PNode(port=0)
        assert node.port == 0
        assert node.node_id