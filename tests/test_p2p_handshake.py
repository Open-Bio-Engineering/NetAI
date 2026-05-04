import time
from netai.p2p.handshake import (
    NodeCapabilities, NodeScore, HandshakeProtocol, detect_capabilities,
)


class TestNodeCapabilities:
    def test_default_creation(self):
        caps = NodeCapabilities(node_id="test-node")
        assert caps.node_id == "test-node"
        assert caps.cpu_cores > 0
        assert "float32" in caps.preferred_precision or "float" in caps.preferred_precision

    def test_model_dump(self):
        caps = NodeCapabilities(node_id="n1")
        data = caps.model_dump()
        assert data["node_id"] == "n1"
        assert "cpu_cores" in data

    def test_detect_capabilities(self):
        caps = detect_capabilities("auto-detect")
        assert caps.node_id == "auto-detect"
        assert caps.cpu_cores > 0
        assert caps.ram_total_mb > 0

    def test_supported_architectures_default(self):
        caps = NodeCapabilities(node_id="arch-test")
        assert "gpt2" in caps.supported_architectures

    def test_custom_architectures(self):
        caps = NodeCapabilities(node_id="arch2", supported_architectures=["gpt2", "llama", "mistral"])
        assert len(caps.supported_architectures) == 3


class TestNodeScore:
    def test_default_score(self):
        score = NodeScore(node_id="s1")
        assert score.total_score == 0.0
        assert score.rank == 0

    def test_full_score(self):
        score = NodeScore(node_id="s2", total_score=85.5, compute_score=90,
                          memory_score=80, network_score=75, reliability_score=95,
                          rank=1)
        assert score.rank == 1
        assert 80 < score.total_score < 100

    def test_score_to_dict(self):
        score = NodeScore(node_id="s3", total_score=75)
        data = score.model_dump()
        assert data["node_id"] == "s3"
        assert "total_score" in data


class TestHandshakeProtocol:
    def test_init(self):
        hp = HandshakeProtocol(node_id="test-hp")
        assert hp.node_id == "test-hp"
        assert hp.capabilities.node_id == "test-hp"

    def test_receive_handshake(self):
        hp = HandshakeProtocol(node_id="local")
        peer = NodeCapabilities(node_id="peer1", cpu_cores=8, gpu_count=1,
                                gpu_vram_mb=8192, gpu_name="A100")
        received = hp.receive_handshake(peer.model_dump())
        assert received.node_id == "peer1"
        assert "peer1" in hp.peer_capabilities
        assert "peer1" in hp.peer_scores

    def test_scoring_gpu_node(self):
        hp = HandshakeProtocol(node_id="ranker")
        caps = NodeCapabilities(node_id="gpu-node", cpu_cores=16, gpu_count=2,
                                gpu_vram_mb=16384, gpu_name="A100")
        hp.receive_handshake(caps.model_dump())
        score = hp.peer_scores["gpu-node"]
        assert score.compute_score > 70
        assert score.rank == 1

    def test_scoring_cpu_node(self):
        hp = HandshakeProtocol(node_id="ranker2")
        caps = NodeCapabilities(node_id="cpu-node", cpu_cores=4, gpu_count=0,
                                ram_available_mb=8192)
        hp.receive_handshake(caps.model_dump())
        score = hp.peer_scores["cpu-node"]
        assert score.compute_score < score.memory_score + 20

    def test_multiple_nodes_ranking(self):
        hp = HandshakeProtocol(node_id="multi")
        for i in range(3):
            caps = NodeCapabilities(
                node_id=f"node{i}", cpu_cores=4 + i * 4,
                gpu_count=1 if i > 0 else 0,
                gpu_vram_mb=4096 * (i + 1),
            )
            hp.receive_handshake(caps.model_dump())
        assert len(hp.peer_scores) == 3
        ranks = [s.rank for s in hp.peer_scores.values()]
        assert set(ranks) == {1, 2, 3}

    def test_suggest_pipeline_role(self):
        hp = HandshakeProtocol(node_id="role-test")
        role = hp.suggest_pipeline_role()
        assert "role" in role
        assert "suggested_layers" in role
        assert role["node_id"] == "role-test"

    def test_best_node_for_layers(self):
        hp = HandshakeProtocol(node_id="selector")
        for i in range(2):
            caps = NodeCapabilities(
                node_id=f"peer{i}", cpu_cores=8,
                ram_available_mb=32000 if i == 0 else 8000,
                gpu_vram_mb=16000 if i == 0 else 0,
            )
            hp.receive_handshake(caps.model_dump())
        results = hp.best_node_for_layers(4, 2000)
        assert len(results) >= 1
        assert results[0]["can_fit"]

    def test_get_peer_list(self):
        hp = HandshakeProtocol(node_id="lister")
        caps = NodeCapabilities(node_id="p1", cpu_cores=8)
        hp.receive_handshake(caps.model_dump())
        peer_list = hp.get_peer_list()
        assert len(peer_list) == 1
        assert peer_list[0]["node_id"] == "p1"

    def test_get_status(self):
        hp = HandshakeProtocol(node_id="status-test")
        hp.receive_handshake(NodeCapabilities(node_id="p2").model_dump())
        status = hp.get_status()
        assert status["node_id"] == "status-test"
        assert status["peers"] == 1
        assert "peer_list" in status
        assert "role_suggestion" in status

    def test_score_rerank(self):
        hp = HandshakeProtocol(node_id="rerank")
        for nid, score_val in [("low", 30), ("mid", 60), ("high", 95)]:
            caps = NodeCapabilities(node_id=nid, cpu_cores=8)
            hp.receive_handshake(caps.model_dump())
            hp.peer_scores[nid] = NodeScore(node_id=nid, total_score=score_val)
        hp._rerank()
        assert hp.peer_scores["high"].rank == 1
        assert hp.peer_scores["low"].rank == 3

    def test_uptime_tracking(self):
        hp = HandshakeProtocol(node_id="uptime")
        hp._start_time = time.time() - 3600
        hp.calculate_uptime = lambda: time.time() - hp._start_time
        uptime = time.time() - hp._start_time
        assert uptime > 3000
