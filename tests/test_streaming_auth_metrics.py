"""Tests for WebSocket streaming, SSE, authentication, Prometheus metrics, and gradient sync."""

import json
import asyncio
import pytest
import numpy as np
from fastapi.testclient import TestClient
from netai.api.app import create_app
from netai.p2p.network import P2PNode
from netai.training.voting import VotingEngine
from netai.training.groups import GroupManager
from netai.scheduler.scheduler import JobScheduler
from netai.inference.engine import InferenceEngine, InferenceRequest, ModelServeConfig
from netai.inference.router import InferenceLoadBalancer, InferenceGateway, RoutingStrategy, InferenceNode
from netai.inference.kv_cache import KVCacheManager, DistributedKVCache
from netai.training.engine import GradientSyncServer, GradientCompressor
from netai.security import SecurityMiddleware, Scope, UserRole


def _make_sec():
    sec = SecurityMiddleware()
    sec.register_user("test-user", "testpassword123", UserRole.ADMIN,
                      scopes=[s.value for s in Scope])
    for ep in ["/api/training/submit", "/api/training/start", "/api/training/stop",
               "/api/inference/load", "/api/inference/run", "/api/inference/unload",
               "/api/jack-in", "/api/training/gradient-sync", "/api/training/gradient-push",
               "/api/training/gradient-pull", "/api/training/gradient-aggregate",
               "/api/training/gradient-peer", "/api/training/gradient-status",
               "/api/vote/propose-model", "/api/vote/cast",
               "/api/pledge", "/api/group/create", "/api/group/join",
               "/api/scheduler/submit",
               "/api/inference/node/register", "/api/inference/stream",
               "/api/inference/stream-sse"]:
        sec.register_public_endpoint(ep)
    return sec


@pytest.fixture
def client():
    app = create_app(
        p2p_node=P2PNode(port=0, node_id="test-node"),
        voting_engine=VotingEngine(),
        group_manager=GroupManager(),
        scheduler=JobScheduler(),
        security=_make_sec(),
    )
    return TestClient(app)


@pytest.fixture
def app_with_inf():
    p2p = P2PNode(port=0, node_id="inf-test-node")
    inf_engine = InferenceEngine(node_id=p2p.node_id)
    inf_lb = InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)
    inf_gw = InferenceGateway(inf_engine, inf_lb)
    app = create_app(
        p2p_node=p2p,
        voting_engine=VotingEngine(),
        group_manager=GroupManager(),
        scheduler=JobScheduler(),
        inference_gateway=inf_gw,
    )
    return app, inf_engine, inf_gw


class TestAuthEndpoints:
    def test_register_user(self, client):
        r = client.post("/api/auth/register", json={
            "user_id": "new-user",
            "password": "testpassword123",
            "role": "user",
        })
        assert r.status_code == 200
        d = r.json()
        assert d["user_id"] == "new-user"
        assert d["role"] == "user"

    def test_login(self, client):
        r = client.post("/api/auth/login", json={
            "user_id": "test-user",
            "password": "testpassword123",
        })
        assert r.status_code == 200
        d = r.json()
        assert "access_token" in d
        assert "refresh_token" in d
        assert d["user_id"] == "test-user"

    def test_login_wrong_password(self, client):
        r = client.post("/api/auth/login", json={
            "user_id": "test-user",
            "password": "wrongpassword",
        })
        assert r.status_code == 401

    def test_create_auth_token_with_auth(self, client):
        login_r = client.post("/api/auth/login", json={
            "user_id": "test-user",
            "password": "testpassword123",
        })
        token = login_r.json()["access_token"]
        r = client.post("/api/auth/token", json={
            "user_id": "test-user",
            "scopes": ["read", "write"],
            "ttl_hours": 12.0,
        }, headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        d = r.json()
        assert "token" in d
        assert d["expires_in_hours"] == 12.0
        assert "write" in d["scopes"]

    def test_verify_valid_token(self, client):
        login_r = client.post("/api/auth/login", json={
            "user_id": "test-user",
            "password": "testpassword123",
        })
        token = login_r.json()["access_token"]
        vr = client.get(f"/api/auth/verify?token={token}")
        assert vr.status_code == 200
        d = vr.json()
        assert d["valid"] is True
        assert d["user_id"] == "test-user"

    def test_verify_invalid_token(self, client):
        vr = client.get("/api/auth/verify?token=nonexistent_token_12345")
        assert vr.status_code == 401

    def test_create_api_key_with_auth(self, client):
        login_r = client.post("/api/auth/login", json={
            "user_id": "test-user",
            "password": "testpassword123",
        })
        token = login_r.json()["access_token"]
        r = client.post("/api/auth/api-key", json={
            "user_id": "test-user",
            "name": "test-key",
            "scopes": ["read", "write", "admin"],
        }, headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        d = r.json()
        assert d["api_key"].startswith("nx_")
        assert d["name"] == "test-key"

    def test_token_expiry(self, client):
        login_r = client.post("/api/auth/login", json={
            "user_id": "test-user",
            "password": "testpassword123",
        })
        token = login_r.json()["access_token"]
        r = client.post("/api/auth/token", json={
            "user_id": "test-user",
            "scopes": ["read"],
            "ttl_hours": 0.00001,
        }, headers={"Authorization": f"Bearer {token}"})
        short_token = r.json()["token"]
        import time
        time.sleep(0.1)
        vr = client.get(f"/api/auth/verify?token={short_token}")
        assert vr.status_code == 401

    def test_multiple_scopes(self, client):
        login_r = client.post("/api/auth/login", json={
            "user_id": "test-user",
            "password": "testpassword123",
        })
        token = login_r.json()["access_token"]
        r = client.post("/api/auth/token", json={
            "user_id": "test-user",
            "scopes": ["read", "write", "admin", "inference", "training"],
            "ttl_hours": 1.0,
        }, headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        d = r.json()
        assert len(d["scopes"]) == 5

    def test_api_key_auth(self, client):
        login_r = client.post("/api/auth/login", json={
            "user_id": "test-user",
            "password": "testpassword123",
        })
        token = login_r.json()["access_token"]
        key_r = client.post("/api/auth/api-key", json={
            "user_id": "test-user",
            "name": "test",
            "scopes": ["read", "write"],
        }, headers={"Authorization": f"Bearer {token}"})
        api_key = key_r.json()["api_key"]
        vr = client.get("/api/auth/verify?token=nonexistent",
                        headers={"X-API-Key": api_key})

    def test_unauthorized_access(self, client):
        r = client.get("/api/security/status")
        assert r.status_code == 401


class TestPrometheusMetrics:
    def test_metrics_endpoint(self, client):
        r = client.get("/api/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers.get("content-type", "")
        text = r.text
        assert "nx_requests_total" in text
        assert "nx_uptime_seconds" in text
        assert "nx_cpu_cores_total" in text
        assert "nx_gpu_count" in text
        assert "nx_ram_total_gb" in text
        assert "nx_peers_connected" in text
        assert "nx_training_jobs_active" in text
        assert "nx_inferences_total" in text
        assert "nx_inference_models_loaded" in text
        assert "nx_groups_total" in text
        assert "nx_proposals_total" in text
        assert "nx_pledges_total" in text
        assert "nx_inference_nodes_total" in text
        assert "nx_inference_nodes_available" in text

    def test_metrics_values_are_numeric(self, client):
        r = client.get("/api/metrics")
        text = r.text
        for line in text.strip().split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            assert len(parts) == 2, f"Bad metric line: {line}"
            name_val = parts[0]
            val = parts[1]
            try:
                float(val.replace("}", "").split()[-1] if "}" in val else val)
            except ValueError:
                if "{" not in name_val:
                    raise AssertionError(f"Non-numeric metric value: {line}")

    def test_metrics_increments_requests(self, client):
        r1 = client.get("/api/metrics")
        count_line1 = [l for l in r1.text.split("\n") if l.startswith("nx_requests_total ") and "#" not in l]
        r2 = client.get("/api/metrics")
        count_line2 = [l for l in r2.text.split("\n") if l.startswith("nx_requests_total ") and "#" not in l]
        if count_line1 and count_line2:
            v1 = int(count_line1[0].split()[-1])
            v2 = int(count_line2[0].split()[-1])
            assert v2 >= v1


class TestStreamingInference:
    def test_stream_infer_local_engine(self):
        import asyncio
        engine = InferenceEngine(node_id="stream-test")
        
        async def run():
            await engine.start()
            config = ModelServeConfig(model_id="test-model", model_name="test-model")
            await engine.load_model(config)
            request = InferenceRequest(model_id="test-model", prompt="Hello", max_tokens=20)
            chunks = []
            async for chunk in engine.stream_infer(request):
                chunks.append(chunk)
            await engine.stop()
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) > 0
        first = chunks[0]
        assert first["type"] == "start"
        assert first["model_id"] == "test-model"
        assert "request_id" in first

        token_chunks = [c for c in chunks if c["type"] == "token"]
        assert len(token_chunks) > 0
        assert all("index" in c for c in token_chunks)

        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert len(done_chunks) == 1
        done = done_chunks[0]
        assert "tokens_generated" in done
        assert "latency_ms" in done
        assert "tokens_per_second" in done
        assert "usage" in done

    def test_stream_infer_model_not_loaded(self):
        import asyncio
        engine = InferenceEngine(node_id="stream-test-2")

        async def run():
            await engine.start()
            request = InferenceRequest(model_id="nonexistent", prompt="test")
            chunks = []
            async for chunk in engine.stream_infer(request):
                chunks.append(chunk)
            await engine.stop()
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) == 1
        assert chunks[0]["type"] == "error"
        assert "not loaded" in chunks[0]["error"]

    def test_stream_infer_engine_not_running(self):
        import asyncio
        engine = InferenceEngine(node_id="stream-test-3")

        async def run():
            request = InferenceRequest(model_id="test", prompt="test")
            chunks = []
            async for chunk in engine.stream_infer(request):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) == 1
        assert chunks[0]["type"] == "error"
        assert "not running" in chunks[0]["error"]


class TestGatewayStreaming:
    def test_gateway_stream_serve_local(self):
        import asyncio
        engine = InferenceEngine(node_id="gw-stream-test")
        lb = InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)
        gw = InferenceGateway(engine, lb)

        async def run():
            await gw.start()
            config = ModelServeConfig(model_id="gw-model", model_name="gw-model")
            await engine.load_model(config)
            request = InferenceRequest(model_id="gw-model", prompt="test prompt", max_tokens=15)
            chunks = []
            async for chunk in gw.stream_serve(request):
                chunks.append(chunk)
            await gw.stop()
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) > 0
        start_chunks = [c for c in chunks if c["type"] == "start"]
        assert len(start_chunks) == 1

    def test_gateway_stream_serve_not_running(self):
        import asyncio
        engine = InferenceEngine(node_id="gw-stream-test-2")
        lb = InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)
        gw = InferenceGateway(engine, lb)

        async def run():
            request = InferenceRequest(model_id="x", prompt="test")
            chunks = []
            async for chunk in gw.stream_serve(request):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) == 1
        assert chunks[0]["type"] == "error"

    def test_gateway_stream_serve_model_not_available(self):
        import asyncio
        engine = InferenceEngine(node_id="gw-stream-test-3")
        lb = InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)
        gw = InferenceGateway(engine, lb)

        async def run():
            await gw.start()
            request = InferenceRequest(model_id="nonexistent-model", prompt="test")
            chunks = []
            async for chunk in gw.stream_serve(request):
                chunks.append(chunk)
            await gw.stop()
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) == 1
        assert chunks[0]["type"] == "error"


class TestSSEInferenceEndpoint:
    def test_sse_inference_stream_needs_model_loaded(self, client):
        r = client.get("/api/inference/stream?model_id=nonexistent&prompt=hello")
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_sse_endpoint_exists(self, client):
        r = client.get("/api/inference/stream?model_id=test&prompt=test&max_tokens=10")
        assert r.status_code == 200


class TestWebSocketEndpoint:
    def test_ws_endpoint_exists(self, client):
        login_r = client.post("/api/auth/login", json={
            "user_id": "test-user", "password": "testpassword123",
        })
        token = login_r.json()["access_token"]
        with client.websocket_connect(f"/ws/inference/stream?token={token}") as ws:
            ws.send_json({"model_id": "test", "prompt": "hello", "max_tokens": 10})
            msg = ws.receive_json()
            assert "type" in msg

    def test_ws_stream_model_not_loaded(self, client):
        login_r = client.post("/api/auth/login", json={
            "user_id": "test-user", "password": "testpassword123",
        })
        token = login_r.json()["access_token"]
        with client.websocket_connect(f"/ws/inference/stream?token={token}") as ws:
            ws.send_json({"model_id": "nonexistent", "prompt": "test"})
            msg = ws.receive_json()
            assert msg["type"] in ("error", "start")


class TestInferenceLoadAndServe:
    def test_load_and_run_inference(self, client):
        r = client.post("/api/inference/load", json={"model_name": "gpt2-small"})
        assert r.status_code == 200
        d = r.json()
        assert "model_id" in d

        r2 = client.post("/api/inference/run", json={
            "model_id": d["model_id"],
            "prompt": "test prompt",
            "max_tokens": 10,
        })
        assert r2.status_code == 200
        result = r2.json()
        assert "request_id" in result

    def test_inference_status(self, client):
        r = client.get("/api/inference/status")
        assert r.status_code == 200
        d = r.json()
        assert "gateway" in d
        assert "local" in d
        assert "cluster" in d

    def test_inference_models_empty(self, client):
        r = client.get("/api/inference/models")
        assert r.status_code == 200

    def test_inference_cache(self, client):
        r = client.get("/api/inference/cache")
        assert r.status_code == 200

    def test_register_inference_node(self, client):
        r = client.post("/api/inference/node/register", json={
            "node_id": "test-inf-node",
            "endpoint": "http://localhost:8080",
            "models_loaded": ["gpt2-small"],
            "gpu_count": 1,
            "cpu_cores": 4,
        })
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestGradientSyncServer:
    def test_init(self):
        gs = GradientSyncServer(node_id="node-1")
        assert gs.node_id == "node-1"
        assert len(gs.peer_endpoints) == 0

    def test_add_remove_peer(self):
        gs = GradientSyncServer(node_id="node-1")
        gs.add_peer("node-2", "http://node2:8080")
        assert "node-2" in gs.peer_endpoints
        gs.remove_peer("node-2")
        assert "node-2" not in gs.peer_endpoints

    def test_push_and_pull_gradients(self):
        gs = GradientSyncServer(node_id="node-1")
        gradients = {
            "layer_0.attn.q_weight": np.random.randn(4, 4).astype(np.float32),
            "layer_0.ffn.up_weight": np.random.randn(4, 8).astype(np.float32),
        }
        asyncio.run(gs.push_gradients("job-1", 1, gradients))
        asyncio.run(gs.aggregate_for_step("job-1", 1))
        result = asyncio.run(gs.pull_aggregated("job-1", 1))
        assert result is not None
        assert "layer_0.attn.q_weight" in result
        assert result["layer_0.attn.q_weight"].shape == (4, 4)

    def test_pull_nonexistent(self):
        gs = GradientSyncServer(node_id="node-1")
        result = asyncio.run(gs.pull_aggregated("nonexistent", 999))
        assert result is None

    def test_aggregate_gradients(self):
        gs = GradientSyncServer(node_id="node-1")
        grad1 = {"layer_0.weight": np.ones((2, 2), dtype=np.float32) * 2.0}
        grad2 = {"layer_0.weight": np.ones((2, 2), dtype=np.float32) * 4.0}
        asyncio.run(gs.push_gradients("job-agg", 1, grad1, node_id="node-1"))
        asyncio.run(gs.push_gradients("job-agg", 1, grad2, node_id="node-2"))
        agg = asyncio.run(gs.aggregate_for_step("job-agg", 1))
        assert "layer_0.weight" in agg
        expected = np.ones((2, 2), dtype=np.float32) * 3.0
        np.testing.assert_array_almost_equal(agg["layer_0.weight"], expected)

    def test_receive_gradients_payload(self):
        gs = GradientSyncServer(node_id="node-1")
        grad = np.random.randn(4, 4).astype(np.float32)
        compressed = GradientCompressor.compress(grad, "topk", 0.5)
        payload = {
            "job_id": "job-rx",
            "step": 5,
            "node_id": "remote-node",
            "gradients": {
                "layer_0.weight": {
                    "shape": [4, 4],
                    "compressed": compressed,
                    "hash": "abc123",
                },
            },
        }
        ok = asyncio.run(gs.receive_gradients(payload))
        assert ok is True

    def test_get_sync_status(self):
        gs = GradientSyncServer(node_id="node-1")
        gs.add_peer("node-2", "http://node2:8080")
        status = gs.get_sync_status()
        assert status["node_id"] == "node-1"
        assert status["peers"] == 1
        assert status["running"] is False


class TestGradientSyncAPI:
    def test_gradient_sync_endpoint(self, client):
        r = client.post("/api/training/gradient-sync", json={
            "job_id": "job-test",
            "step": 1,
            "node_id": "remote-node",
            "gradients": {},
        })
        assert r.status_code == 200
        assert r.json()["status"] == "error"

    def test_gradient_status_endpoint(self, client):
        r = client.get("/api/training/gradient-status")
        assert r.status_code == 200
        d = r.json()
        assert "node_id" in d
        assert "peers" in d

    def test_add_gradient_peer(self, client):
        r = client.post("/api/training/gradient-peer?node_id=test-peer&endpoint=http%3A%2F%2Ftest%3A8080")
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "ok"

    def test_gradient_pull_not_found(self, client):
        r = client.get("/api/training/gradient-pull/nonexistent/999")
        assert r.status_code == 404

    def test_gradient_push_not_found(self, client):
        r = client.post("/api/training/gradient-push/nonexistent/1")
        assert r.status_code == 200
        assert r.json()["status"] == "no_gradients"


class TestGradientCompressor:
    def test_topk_compress_decompress(self):
        grad = np.random.randn(100, 100).astype(np.float32)
        compressed = GradientCompressor.compress(grad, "topk", 0.01)
        assert compressed["method"] == "topk"
        decompressed = GradientCompressor.decompress(compressed)
        assert decompressed.shape == grad.shape

    def test_quantize_compress_decompress(self):
        grad = np.random.randn(50, 50).astype(np.float32)
        compressed = GradientCompressor.compress(grad, "quantize")
        assert compressed["method"] == "quantize"
        decompressed = GradientCompressor.decompress(compressed)
        assert decompressed.shape == grad.shape

    def test_none_compress_decompress(self):
        grad = np.random.randn(4, 4).astype(np.float32)
        compressed = GradientCompressor.compress(grad, "none")
        assert compressed["method"] == "none"
        decompressed = GradientCompressor.decompress(compressed)
        np.testing.assert_array_almost_equal(decompressed, grad)