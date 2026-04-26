"""Comprehensive inference validation tests.

Tests every code path through:
- InferenceEngine: load, infer, stream, drain, unload, status, shard creation
- InferenceLoadBalancer: register, unregister, heartbeat, all routing strategies
- InferenceGateway: serve local, serve remote, stream, load_model status
- KVCacheManager: get, put, eviction, TTL, partition, distributed, stats, prefix cache
- DistributedKVCache: peer registration, affinity, aggregate stats
- ModelMirror: register, unregister, nearest, status
- AutoLoader + InferenceEngine integration: load plans -> load model -> infer
- Pipeline-parallel inference: multi-shard, multi-node, model handoff
- Edge cases: empty prompts, max tokens boundaries, concurrent requests
- Performance: latency under load, queue throughput
"""

import asyncio
import json
import time
import pytest
import numpy as np

from netai.inference.engine import (
    InferenceEngine, InferenceRequest, InferenceResponse, InferenceStatus,
    ModelServeConfig, ModelReplica, ModelShard, ShardType, ModelMirror,
    InferenceMetrics,
)
from netai.inference.router import (
    InferenceLoadBalancer, InferenceGateway, InferenceNode, RoutingStrategy,
)
from netai.inference.kv_cache import (
    KVCacheManager, DistributedKVCache, CacheEntry, CachePartition,
)
from netai.inference.autoloader import (
    AutoLoader, ModelEntry, ModelRegistry, ModelSizeClass,
)
from pathlib import Path


# ============================================================
# InferenceEngine deep validation
# ============================================================

class TestInferenceEngineDeep:
    @pytest.fixture
    def engine(self):
        return InferenceEngine(node_id="test-node-deep")

    @pytest.mark.asyncio
    async def test_start_sets_ready_status(self, engine):
        assert engine.status == InferenceStatus.IDLE
        await engine.start()
        assert engine.status == InferenceStatus.READY
        assert engine._running is True
        assert engine._request_queue is not None
        await engine.stop()

    @pytest.mark.asyncio
    async def test_stop_sets_offline(self, engine):
        await engine.start()
        await engine.stop()
        assert engine.status == InferenceStatus.OFFLINE
        assert engine._running is False

    @pytest.mark.asyncio
    async def test_load_model_creates_replica_with_shard(self, engine):
        config = ModelServeConfig(model_id="gpt2-small", model_name="GPT-2 Small", num_shards=1)
        replica = await engine.load_model(config)
        assert replica.model_id == "gpt2-small"
        assert replica.status == InferenceStatus.READY
        assert len(replica.shard_ids) == 1
        assert "gpt2-small" in engine.models
        assert "gpt2-small" in engine._model_weights

    @pytest.mark.asyncio
    async def test_load_model_multi_shard(self, engine):
        config = ModelServeConfig(model_id="big-model", model_name="Big Model", num_shards=4)
        replica = await engine.load_model(config)
        assert len(replica.shard_ids) == 4
        shards = engine.shards.get("big-model", {})
        assert len(shards) == 4
        for i, (sid, shard) in enumerate(shards.items()):
            assert shard.shard_index == i
            assert shard.total_shards == 4
            assert shard.status == InferenceStatus.READY
            assert shard.size_mb > 0

    @pytest.mark.asyncio
    async def test_load_model_custom_weights(self, engine):
        weights = {
            "layer0.w": np.random.randn(64, 64).astype(np.float32),
            "layer1.w": np.random.randn(64, 64).astype(np.float32),
        }
        config = ModelServeConfig(model_id="custom-model", model_name="Custom")
        replica = await engine.load_model(config, weights=weights)
        assert replica.model_id == "custom-model"
        assert "layer0.w" in engine._model_weights["custom-model"]

    @pytest.mark.asyncio
    async def test_infer_happy_path(self, engine):
        await engine.start()
        config = ModelServeConfig(model_id="test-model")
        await engine.load_model(config)
        req = InferenceRequest(model_id="test-model", prompt="Hello world")
        resp = await engine.infer(req)
        assert resp.error is None
        assert resp.tokens_generated > 0
        assert resp.latency_ms > 0
        assert resp.text != ""
        assert resp.finish_reason in ("stop", "length")
        assert resp.usage["completion_tokens"] > 0
        await engine.stop()

    @pytest.mark.asyncio
    async def test_infer_not_running(self, engine):
        req = InferenceRequest(model_id="test", prompt="Hello")
        resp = await engine.infer(req)
        assert "not running" in resp.error.lower()

    @pytest.mark.asyncio
    async def test_infer_model_not_loaded(self, engine):
        await engine.start()
        req = InferenceRequest(model_id="nonexistent", prompt="Hello")
        resp = await engine.infer(req)
        assert "not loaded" in resp.error.lower()
        await engine.stop()

    @pytest.mark.asyncio
    async def test_infer_prompt_too_long(self, engine):
        await engine.start()
        config = ModelServeConfig(model_id="test-model")
        await engine.load_model(config)
        req = InferenceRequest(model_id="test-model", prompt="x" * 100001)
        resp = await engine.infer(req)
        assert "too long" in resp.error.lower()
        await engine.stop()

    @pytest.mark.asyncio
    async def test_infer_uses_least_loaded_replica(self, engine):
        await engine.start()
        config = ModelServeConfig(model_id="llm-1", num_replicas=1)
        replica = await engine.load_model(config)
        config2 = ModelServeConfig(model_id="llm-2", num_replicas=1)
        await engine.load_model(config2)
        req = InferenceRequest(model_id="llm-1", prompt="test")
        resp = await engine.infer(req)
        assert resp.error is None
        assert resp.tokens_generated > 0
        await engine.stop()

    @pytest.mark.asyncio
    async def test_infer_metrics_recorded(self, engine):
        await engine.start()
        config = ModelServeConfig(model_id="m-metrics")
        await engine.load_model(config)
        req = InferenceRequest(model_id="m-metrics", prompt="Metrics test")
        await engine.infer(req)
        assert len(engine.metrics_history) == 1
        m = engine.metrics_history[0]
        assert m.model_id == "m-metrics"
        assert m.latency_ms > 0
        assert m.completion_tokens > 0
        await engine.stop()

    @pytest.mark.asyncio
    async def test_drain_model(self, engine):
        await engine.start()
        config = ModelServeConfig(model_id="drain-model")
        await engine.load_model(config)
        result = await engine.drain_model("drain-model", timeout=0.5)
        assert result is True
        await engine.stop()

    @pytest.mark.asyncio
    async def test_drain_nonexistent_model(self, engine):
        await engine.start()
        result = await engine.drain_model("nonexistent", timeout=0.5)
        assert result is True
        await engine.stop()

    @pytest.mark.asyncio
    async def test_unload_removes_everything(self, engine):
        await engine.start()
        config = ModelServeConfig(model_id="unload-test")
        await engine.load_model(config)
        assert "unload-test" in engine.models
        assert "unload-test" in engine._model_weights
        result = await engine.unload_model("unload-test")
        assert result is True
        assert "unload-test" not in engine.models
        assert "unload-test" not in engine._model_weights
        assert "unload-test" not in engine.replicas
        assert "unload-test" not in engine.shards
        await engine.stop()

    @pytest.mark.asyncio
    async def test_stream_infer(self, engine):
        await engine.start()
        config = ModelServeConfig(model_id="stream-model")
        await engine.load_model(config)
        req = InferenceRequest(model_id="stream-model", prompt="Stream test", max_tokens=20)
        chunks = []
        async for chunk in engine.stream_infer(req):
            chunks.append(chunk)
        assert len(chunks) >= 2
        assert chunks[0]["type"] == "start"
        assert chunks[-1]["type"] == "done"
        assert chunks[-1]["tokens_generated"] > 0
        assert chunks[-1]["latency_ms"] > 0
        await engine.stop()

    @pytest.mark.asyncio
    async def test_stream_infer_not_running(self, engine):
        req = InferenceRequest(model_id="test", prompt="Hello")
        chunks = []
        async for chunk in engine.stream_infer(req):
            chunks.append(chunk)
        assert len(chunks) >= 1
        assert chunks[0]["type"] == "error"

    @pytest.mark.asyncio
    async def test_stream_infer_model_not_loaded(self, engine):
        await engine.start()
        req = InferenceRequest(model_id="nonexistent", prompt="Hello")
        chunks = []
        async for chunk in engine.stream_infer(req):
            chunks.append(chunk)
        assert len(chunks) >= 1
        assert chunks[0]["type"] == "error"
        await engine.stop()

    @pytest.mark.asyncio
    async def test_get_status(self, engine):
        await engine.start()
        config = ModelServeConfig(model_id="status-model", model_name="Status Model")
        await engine.load_model(config)
        req = InferenceRequest(model_id="status-model", prompt="status")
        await engine.infer(req)
        status = engine.get_status()
        assert status["node_id"] == "test-node-deep"
        assert status["models_loaded"] == 1
        assert "status-model" in status["models"]
        info = status["models"]["status-model"]
        assert info["total_inferences"] >= 1
        assert info["available_replicas"] >= 1
        await engine.stop()

    @pytest.mark.asyncio
    async def test_multiple_loads_same_model(self, engine):
        config = ModelServeConfig(model_id="dup-model")
        r1 = await engine.load_model(config)
        r2 = await engine.load_model(config)
        assert r1.replica_id != r2.replica_id
        assert len(engine.replicas["dup-model"]) == 2

    @pytest.mark.asyncio
    async def test_concurrent_inferences(self, engine):
        await engine.start()
        config = ModelServeConfig(model_id="concurrent-model")
        await engine.load_model(config)
        tasks = []
        for i in range(10):
            req = InferenceRequest(model_id="concurrent-model", prompt=f"Request {i}")
            tasks.append(engine.infer(req))
        results = await asyncio.gather(*tasks)
        errors = [r for r in results if r.error is not None]
        assert len(errors) == 0
        assert all(r.tokens_generated > 0 for r in results)
        await engine.stop()

    @pytest.mark.asyncio
    async def test_shard_weights_size_mb(self, engine):
        config = ModelServeConfig(model_id="shard-size", num_shards=3)
        replica = await engine.load_model(config)
        shards = engine.shards["shard-size"]
        total_size = sum(s.size_mb for s in shards.values())
        assert total_size > 0

    @pytest.mark.asyncio
    async def test_inference_request_defaults(self, engine):
        req = InferenceRequest()
        assert req.max_tokens == 256
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.top_k == 50
        assert req.stream is False
        assert req.priority == 0
        assert req.timeout_ms == 30000
        assert req.request_id != ""


# ============================================================
# InferenceLoadBalancer (router) deep validation
# ============================================================

class TestLoadBalancerDeep:
    @pytest.fixture
    def lb(self):
        return InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)

    def test_register_node(self, lb):
        node = InferenceNode(node_id="n1", endpoint="http://n1:8001", models_loaded=["gpt2"])
        lb.register_node(node)
        assert "n1" in lb.nodes
        assert "gpt2" in lb.model_nodes
        assert "n1" in lb.model_nodes["gpt2"]

    def test_register_multiple_nodes_same_model(self, lb):
        n1 = InferenceNode(node_id="n1", endpoint="http://n1:8001", models_loaded=["gpt2"])
        n2 = InferenceNode(node_id="n2", endpoint="http://n2:8001", models_loaded=["gpt2"])
        lb.register_node(n1)
        lb.register_node(n2)
        assert len(lb.model_nodes["gpt2"]) == 2

    def test_register_duplicate_node_id(self, lb):
        n1 = InferenceNode(node_id="n1", endpoint="http://n1:8001", models_loaded=["gpt2"])
        n1_updated = InferenceNode(node_id="n1", endpoint="http://n1:8002", models_loaded=["gpt2", "llama"])
        lb.register_node(n1)
        lb.register_node(n1_updated)
        assert lb.nodes["n1"].endpoint == "http://n1:8002"

    def test_unregister_node(self, lb):
        n1 = InferenceNode(node_id="n1", endpoint="http://n1:8001", models_loaded=["gpt2"])
        lb.register_node(n1)
        lb.unregister_node("n1")
        assert "n1" not in lb.nodes

    def test_unregister_nonexistent(self, lb):
        lb.unregister_node("n999")
        assert True

    def test_update_heartbeat(self, lb):
        n1 = InferenceNode(node_id="n1", endpoint="http://n1:8001", models_loaded=["gpt2"])
        lb.register_node(n1)
        old_hb = n1.last_heartbeat
        time.sleep(0.01)
        lb.update_node_heartbeat("n1", load=5, avg_latency=50.0)
        assert n1.last_heartbeat > old_hb
        assert n1.current_load == 5
        assert n1.avg_latency_ms == 50.0

    def test_heartbeat_improves_health(self, lb):
        n1 = InferenceNode(node_id="n1", endpoint="http://n1:8001", models_loaded=["gpt2"], health_score=0.5)
        lb.register_node(n1)
        lb.update_node_heartbeat("n1", load=0, avg_latency=10.0)
        assert n1.health_score > 0.5
        assert n1.health_score <= 1.0

    def test_heartbeat_nonexistent_node(self, lb):
        lb.update_node_heartbeat("n999", load=0, avg_latency=10.0)
        assert True

    def test_add_model_to_node(self, lb):
        n1 = InferenceNode(node_id="n1", endpoint="http://n1:8001", models_loaded=["gpt2"])
        lb.register_node(n1)
        lb.add_model_to_node("n1", "llama")
        assert "llama" in lb.nodes["n1"].models_loaded
        assert "n1" in lb.model_nodes["llama"]

    def test_add_model_registers_mirror(self, lb):
        n1 = InferenceNode(node_id="n1", endpoint="http://n1:8001", models_loaded=[])
        lb.register_node(n1)
        lb.add_model_to_node("n1", "gpt2")
        mirrors = lb.mirror.get_mirrors("gpt2")
        assert "n1" in mirrors

    def test_route_round_robin(self):
        lb = InferenceLoadBalancer(RoutingStrategy.ROUND_ROBIN)
        for i in range(4):
            n = InferenceNode(node_id=f"n{i}", endpoint=f"http://n{i}:8001",
                              models_loaded=["gpt2"], status=InferenceStatus.READY)
            lb.register_node(n)
        req = InferenceRequest(model_id="gpt2", prompt="test")
        results = [lb.route_request(req) for _ in range(4)]
        assert len(set(results)) >= 2

    def test_route_least_loaded(self):
        lb = InferenceLoadBalancer(RoutingStrategy.LEAST_LOADED)
        n1 = InferenceNode(node_id="n1", endpoint="http://n1:8001", models_loaded=["gpt2"],
                          status=InferenceStatus.READY, current_load=5)
        n2 = InferenceNode(node_id="n2", endpoint="http://n2:8001", models_loaded=["gpt2"],
                          status=InferenceStatus.READY, current_load=1)
        lb.register_node(n1)
        lb.register_node(n2)
        req = InferenceRequest(model_id="gpt2", prompt="test")
        result = lb.route_request(req)
        assert result == "n2"

    def test_route_lowest_latency(self):
        lb = InferenceLoadBalancer(RoutingStrategy.LOWEST_LATENCY)
        n1 = InferenceNode(node_id="n1", endpoint="http://n1:8001", models_loaded=["gpt2"],
                          status=InferenceStatus.READY, avg_latency_ms=100.0)
        n2 = InferenceNode(node_id="n2", endpoint="http://n2:8001", models_loaded=["gpt2"],
                          status=InferenceStatus.READY, avg_latency_ms=20.0)
        lb.register_node(n1)
        lb.register_node(n2)
        req = InferenceRequest(model_id="gpt2", prompt="test")
        result = lb.route_request(req)
        assert result == "n2"

    def test_route_hash_based_deterministic(self):
        lb = InferenceLoadBalancer(RoutingStrategy.HASH_BASED)
        for i in range(3):
            n = InferenceNode(node_id=f"n{i}", endpoint=f"http://n{i}:8001",
                              models_loaded=["gpt2"], status=InferenceStatus.READY)
            lb.register_node(n)
        req = InferenceRequest(model_id="gpt2", prompt="test", user_id="user1")
        r1 = lb.route_request(req)
        r2 = lb.route_request(req)
        assert r1 == r2

    def test_route_random(self):
        lb = InferenceLoadBalancer(RoutingStrategy.RANDOM)
        for i in range(3):
            n = InferenceNode(node_id=f"n{i}", endpoint=f"http://n{i}:8001",
                              models_loaded=["gpt2"], status=InferenceStatus.READY)
            lb.register_node(n)
        req = InferenceRequest(model_id="gpt2", prompt="test")
        results = set()
        for _ in range(30):
            r = lb.route_request(req)
            if r:
                results.add(r)
        assert len(results) >= 2

    def test_route_adaptive_prefers_low_load_gpu(self):
        lb = InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)
        n1 = InferenceNode(node_id="n1", endpoint="http://n1:8001", models_loaded=["gpt2"],
                          status=InferenceStatus.READY, current_load=80, capacity=100,
                          gpu_count=1, avg_latency_ms=50.0)
        n2 = InferenceNode(node_id="n2", endpoint="http://n2:8001", models_loaded=["gpt2"],
                          status=InferenceStatus.READY, current_load=10, capacity=100,
                          gpu_count=1, avg_latency_ms=10.0)
        lb.register_node(n1)
        lb.register_node(n2)
        req = InferenceRequest(model_id="gpt2", prompt="test")
        result = lb.route_request(req)
        assert result == "n2"

    def test_route_no_candidates(self, lb):
        req = InferenceRequest(model_id="nonexistent", prompt="test")
        result = lb.route_request(req)
        assert result is None

    def test_route_group_id_filter(self, lb):
        n1 = InferenceNode(node_id="n1", endpoint="http://n1:8001", models_loaded=["gpt2"],
                          status=InferenceStatus.READY, group_id="team-a")
        n2 = InferenceNode(node_id="n2", endpoint="http://n2:8001", models_loaded=["gpt2"],
                          status=InferenceStatus.READY, group_id="team-b")
        lb.register_node(n1)
        lb.register_node(n2)
        req = InferenceRequest(model_id="gpt2", prompt="test", group_id="team-a")
        result = lb.route_request(req)
        assert result == "n1"

    def test_node_at_capacity_not_available(self):
        n = InferenceNode(node_id="n1", endpoint="http://n1:8001", models_loaded=["gpt2"],
                         status=InferenceStatus.READY, current_load=100, capacity=100)
        assert n.is_available is False

    def test_node_low_health_not_available(self):
        n = InferenceNode(node_id="n1", endpoint="http://n1:8001", models_loaded=["gpt2"],
                         status=InferenceStatus.READY, health_score=0.1)
        assert n.is_available is False

    def test_node_stale_heartbeat_not_available(self):
        n = InferenceNode(node_id="n1", endpoint="http://n1:8001", models_loaded=["gpt2"],
                         status=InferenceStatus.READY, last_heartbeat=time.time() - 300)
        assert n.is_available is False

    @pytest.mark.asyncio
    async def test_lb_start_stop(self, lb):
        await lb.start()
        assert lb._running is True
        assert lb._request_queue is not None
        await lb.stop()
        assert lb._running is False

    def test_get_status(self, lb):
        n1 = InferenceNode(node_id="n1", endpoint="http://n1:8001", models_loaded=["gpt2"],
                          status=InferenceStatus.READY)
        lb.register_node(n1)
        status = lb.get_status()
        assert status["strategy"] == "adaptive"
        assert status["total_nodes"] == 1
        assert "gpt2" in status["models"]


# ============================================================
# InferenceGateway deep validation
# ============================================================

class TestGatewayDeep:
    @pytest.mark.asyncio
    async def test_gateway_serves_local_model(self):
        engine = InferenceEngine(node_id="gw-local")
        lb = InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)
        gw = InferenceGateway(engine, lb)
        await gw.start()
        config = ModelServeConfig(model_id="local-llm")
        await gw.load_model(config)
        req = InferenceRequest(model_id="local-llm", prompt="Hello gateway")
        resp = await gw.serve(req)
        assert resp.error is None
        assert resp.tokens_generated > 0
        await gw.stop()

    @pytest.mark.asyncio
    async def test_gateway_stream_local(self):
        engine = InferenceEngine(node_id="gw-stream")
        lb = InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)
        gw = InferenceGateway(engine, lb)
        await gw.start()
        config = ModelServeConfig(model_id="stream-llm")
        await gw.load_model(config)
        req = InferenceRequest(model_id="stream-llm", prompt="Stream gateway", max_tokens=15)
        chunks = []
        async for chunk in gw.stream_serve(req):
            chunks.append(chunk)
        assert len(chunks) >= 2
        assert chunks[0]["type"] == "start"
        await gw.stop()

    @pytest.mark.asyncio
    async def test_gateway_model_not_available(self):
        engine = InferenceEngine(node_id="gw-missing")
        lb = InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)
        gw = InferenceGateway(engine, lb)
        await gw.start()
        req = InferenceRequest(model_id="nonexistent", prompt="test")
        resp = await gw.serve(req)
        assert resp.error is not None
        await gw.stop()

    @pytest.mark.asyncio
    async def test_gateway_not_running(self):
        engine = InferenceEngine(node_id="gw-down")
        lb = InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)
        gw = InferenceGateway(engine, lb)
        req = InferenceRequest(model_id="test", prompt="test")
        resp = await gw.serve(req)
        assert "not available" in resp.error.lower() or "not running" in resp.error.lower()

    @pytest.mark.asyncio
    async def test_gateway_load_model_registers_in_lb(self):
        engine = InferenceEngine(node_id="gw-reg")
        lb = InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)
        engine_node = InferenceNode(node_id="gw-reg", endpoint="http://gw-reg:8001",
                                    models_loaded=[], status=InferenceStatus.READY)
        lb.register_node(engine_node)
        gw = InferenceGateway(engine, lb)
        await gw.start()
        config = ModelServeConfig(model_id="reg-model")
        await gw.load_model(config)
        assert "gw-reg" in lb.model_nodes.get("reg-model", [])
        assert "reg-model" in lb.nodes["gw-reg"].models_loaded
        await gw.stop()

    @pytest.mark.asyncio
    async def test_gateway_status(self):
        engine = InferenceEngine(node_id="gw-status")
        lb = InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)
        gw = InferenceGateway(engine, lb)
        await gw.start()
        status = gw.get_status()
        assert status["gateway"] == "running"
        assert "local" in status
        assert "cluster" in status
        await gw.stop()


# ============================================================
# KV Cache deep validation
# ============================================================

class TestKVCacheDeep:
    def test_put_and_get(self):
        cache = KVCacheManager(max_size_mb=100.0, ttl_seconds=600.0)
        entry = cache.put("gpt2", "Hello world", {"layer0": np.zeros((12, 64))})
        assert entry.cache_id != ""
        assert entry.size_mb > 0
        result = cache.get("gpt2", "Hello world")
        assert result is not None
        assert result.cache_id == entry.cache_id

    def test_put_overwrites_existing(self):
        cache = KVCacheManager(max_size_mb=100.0, ttl_seconds=600.0)
        cache.put("gpt2", "test", {"k": "v1"})
        cache.put("gpt2", "test", {"k": "v2"})
        result = cache.get("gpt2", "test")
        assert result.access_count == 1

    def test_cache_miss(self):
        cache = KVCacheManager()
        result = cache.get("gpt2", "missing prompt")
        assert result is None
        assert cache._misses == 1

    def test_cache_eviction(self):
        cache = KVCacheManager(max_size_mb=0.05, ttl_seconds=600.0)
        for i in range(20):
            data = np.random.randn(100, 100).astype(np.float32)
            cache.put("model", f"prompt-{i}", {"data": data})
        assert cache._evictions > 0
        assert cache._current_size_mb <= cache.max_size_mb

    def test_cache_ttl_expiry(self):
        cache = KVCacheManager(max_size_mb=100.0, ttl_seconds=0.01)
        cache.put("model", "expiring soon", {"k": "v"})
        time.sleep(0.02)
        result = cache.get("model", "expiring soon")
        assert result is None

    def test_cache_stats(self):
        cache = KVCacheManager(max_size_mb=100.0)
        cache.put("m", "p1", {"k": "v1"})
        cache.put("m", "p2", {"k": "v2"})
        cache.get("m", "p1")
        cache.get("m", "missing")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 2
        assert stats["evictions"] == 0

    def test_prefix_cache(self):
        cache = KVCacheManager(max_size_mb=100.0)
        data = [[1.0, 2.0], [3.0, 4.0]]
        entry = cache.put_prefix_cache("m", "system prompt", data)
        assert entry.cache_id != ""
        result = cache.get_prefix_cache("m", "system prompt")
        assert result is not None

    def test_partition(self):
        cache = KVCacheManager()
        p = cache.create_partition("gpt2", "node1", layer_start=0, layer_end=12, max_mb=256.0)
        assert p.model_id == "gpt2"
        assert p.node_id == "node1"
        assert p.layer_start == 0
        assert p.layer_end == 12
        got = cache.get_partition("gpt2", p.partition_id)
        assert got is not None
        assert got.partition_id == p.partition_id

    def test_partition_usage(self):
        cache = KVCacheManager()
        p = cache.create_partition("gpt2", "node1", layer_start=0, layer_end=12, max_mb=256.0)
        assert p.usage_pct == 0.0
        p.size_mb = 128.0
        assert p.usage_pct == 50.0

    def test_clear(self):
        cache = KVCacheManager()
        cache.put("m", "p", {"k": "v"})
        assert len(cache._cache) == 1
        cache.clear()
        assert len(cache._cache) == 0
        assert cache._current_size_mb == 0.0

    def test_distributed_cache_status(self):
        cache = KVCacheManager()
        cache.create_partition("gpt2", "node1", 0, 6)
        cache.create_partition("gpt2", "node2", 6, 12)
        status = cache.get_distributed_cache_status("gpt2")
        assert status["num_partitions"] == 2
        assert "gpt2" in status["model_id"]


# ============================================================
# DistributedKVCache deep validation
# ============================================================

class TestDistributedKVCacheDeep:
    def test_register_peer(self):
        local = KVCacheManager()
        dkc = DistributedKVCache(local, node_id="node-a")
        dkc.register_peer_cache("node-b", {"entries": 10, "size_mb": 50.0, "hit": False})
        assert "node-b" in dkc.peer_caches

    def test_find_cached_request_local_hit(self):
        local = KVCacheManager()
        dkc = DistributedKVCache(local, node_id="node-a")
        local.put("gpt2", "hello", {"k": "v"})
        node_id, cache_id = dkc.find_cached_request("gpt2", "hello")
        assert node_id == "node-a"
        assert cache_id is not None

    def test_find_cached_request_remote_hit(self):
        local = KVCacheManager()
        dkc = DistributedKVCache(local, node_id="node-a")
        dkc.register_peer_cache("node-b", {"entries": 5, "size_mb": 20.0, "hit": True, "cache_id": "abc123"})
        node_id, cache_id = dkc.find_cached_request("gpt2", "not local")
        assert node_id == "node-b"
        assert cache_id == "abc123"

    def test_find_cached_request_miss(self):
        local = KVCacheManager()
        dkc = DistributedKVCache(local, node_id="node-a")
        node_id, cache_id = dkc.find_cached_request("gpt2", "nothing here")
        assert node_id == ""
        assert cache_id is None

    def test_compute_cache_affinity(self):
        local = KVCacheManager()
        dkc = DistributedKVCache(local, node_id="node-a")
        affinity = dkc.compute_cache_affinity("gpt2", "hello world")
        assert affinity in ("node-a", "local")

    def test_aggregate_stats(self):
        local = KVCacheManager()
        dkc = DistributedKVCache(local, node_id="node-a")
        local.put("gpt2", "hello", {"k": "v"})
        dkc.register_peer_cache("node-b", {"entries": 10, "size_mb": 50.0})
        stats = dkc.get_aggregate_stats()
        assert stats["local"]["entries"] == 1
        assert stats["peer_count"] == 1
        assert stats["aggregate_entries"] == 11


# ============================================================
# ModelMirror deep validation
# ============================================================

class TestModelMirrorDeep:
    def test_register_and_get_mirrors(self):
        mirror = ModelMirror()
        assert mirror.register_mirror("gpt2", "node1") is True
        assert mirror.register_mirror("gpt2", "node2") is True
        assert mirror.register_mirror("gpt2", "node1") is False  # duplicate
        assert mirror.get_mirror_count("gpt2") == 2
        mirrors = mirror.get_mirrors("gpt2")
        assert "node1" in mirrors
        assert "node2" in mirrors

    def test_unregister_mirror(self):
        mirror = ModelMirror()
        mirror.register_mirror("gpt2", "node1")
        mirror.register_mirror("gpt2", "node2")
        mirror.unregister_mirror("gpt2", "node1")
        assert mirror.get_mirror_count("gpt2") == 1

    def test_find_nearest_mirror_self(self):
        mirror = ModelMirror()
        mirror.register_mirror("gpt2", "node1")
        mirror.register_mirror("gpt2", "node2")
        result = mirror.find_nearest_mirror("gpt2", "node1")
        assert result == "node1"

    def test_find_nearest_mirror_by_latency(self):
        mirror = ModelMirror()
        mirror.register_mirror("gpt2", "node1")
        mirror.register_mirror("gpt2", "node2")
        latencies = {"node1": 100.0, "node2": 20.0}
        result = mirror.find_nearest_mirror("gpt2", "node-local", latencies)
        assert result == "node2"

    def test_find_nearest_mirror_no_mirrors(self):
        mirror = ModelMirror()
        result = mirror.find_nearest_mirror("nonexistent", "node1")
        assert result is None

    def test_mirror_status(self):
        mirror = ModelMirror()
        mirror.register_mirror("gpt2", "node1")
        mirror.register_mirror("llama", "node2")
        status = mirror.get_status()
        assert status["mirrored_models"] == 2
        assert "gpt2" in status["models"]
        assert "llama" in status["models"]


# ============================================================
# AutoLoader + InferenceEngine integration
# ============================================================

class TestAutoLoaderIntegration:
    @pytest.fixture
    def make_env(self, tmp_path):
        catalog = {
            "models": [
                {"model_id": "mini-1", "name": "Mini 1", "architecture": "llama", "params_m": 50,
                 "vram_required_mb": {"q4_k_m": 300}, "context_length": 4096},
                {"model_id": "small-1", "name": "Small 1", "architecture": "llama", "params_m": 200,
                 "vram_required_mb": {"q4_k_m": 4500}, "context_length": 8192},
                {"model_id": "mid-1", "name": "Mid 1", "architecture": "llama", "params_m": 500,
                 "vram_required_mb": {"q4_k_m": 40000}, "context_length": 131072},
            ]
        }
        (tmp_path / "catalog.json").write_text(json.dumps(catalog))
        reg = ModelRegistry(catalog_path=tmp_path / "catalog.json")
        assert reg.load_local() is True
        return reg

    @pytest.mark.asyncio
    async def test_autoloader_to_engine_flow(self, make_env):
        reg = make_env
        loader = AutoLoader(reg, available_vram_mb=8000, available_nodes=1)
        plan = loader.compute_load_plan()
        assert len(plan) >= 1
        assert plan[0]["model_id"] == "mini-1"

        engine = InferenceEngine(node_id="auto-node")
        await engine.start()
        for item in plan:
            config = ModelServeConfig(
                model_id=item["model_id"],
                model_name=item["name"],
                num_shards=item["config"].get("num_shards", 1),
                quantization=item["quant"],
            )
            replica = await engine.load_model(config)
            assert replica.model_id == item["model_id"]
            assert replica.status == InferenceStatus.READY
            loader.mark_loaded(item["model_id"], item["vram_mb"])

        req = InferenceRequest(model_id=plan[0]["model_id"], prompt="Auto-loaded model test")
        resp = await engine.infer(req)
        assert resp.error is None
        assert resp.tokens_generated > 0

        status = loader.get_status()
        assert len(status["loaded_models"]) >= 1
        await engine.stop()

    @pytest.mark.asyncio
    async def test_autoloader_then_unload(self, make_env):
        reg = make_env
        loader = AutoLoader(reg, available_vram_mb=8000)
        plan = loader.compute_load_plan()
        engine = InferenceEngine(node_id="unload-node")
        await engine.start()
        for item in plan:
            config = ModelServeConfig(model_id=item["model_id"], model_name=item["name"])
            await engine.load_model(config)
            loader.mark_loaded(item["model_id"], item["vram_mb"])

        loaded = loader.get_loaded_models()
        assert len(loaded) >= 1

        for item in plan:
            await engine.unload_model(item["model_id"])
            loader.mark_unloaded(item["model_id"])

        assert len(loader.get_loaded_models()) == 0
        await engine.stop()

    @pytest.mark.asyncio
    async def test_low_vram_loads_mini_only(self, make_env):
        reg = make_env
        loader = AutoLoader(reg, available_vram_mb=500)
        plan = loader.compute_load_plan()
        assert len(plan) >= 1
        assert all(p["size_class"] == "mini" for p in plan)

    @pytest.mark.asyncio
    async def test_forced_model_loading(self, make_env):
        reg = make_env
        loader = AutoLoader(reg, available_vram_mb=8000)
        plan = loader.compute_load_plan(force_models=["small-1"])
        assert len(plan) == 1
        assert plan[0]["model_id"] == "small-1"


# ============================================================
# Pipeline-parallel inference simulation
# ============================================================

class TestPipelineParallel:
    @pytest.mark.asyncio
    async def test_multi_shard_model_inference(self):
        engine = InferenceEngine(node_id="pipeline-node")
        await engine.start()
        config = ModelServeConfig(model_id="pipe-model", num_shards=4, shard_type=ShardType.PIPELINE)
        replica = await engine.load_model(config)
        assert len(replica.shard_ids) == 4

        req = InferenceRequest(model_id="pipe-model", prompt="Pipeline test")
        resp = await engine.infer(req)
        assert resp.error is None
        assert resp.tokens_generated > 0
        await engine.stop()

    @pytest.mark.asyncio
    async def test_pipeline_through_lb_with_multiple_nodes(self):
        lb = InferenceLoadBalancer(RoutingStrategy.LEAST_LOADED)
        engine = InferenceEngine(node_id="main-node")
        gw = InferenceGateway(engine, lb)

        node_configs = [
            InferenceNode(node_id="worker-0", endpoint="http://w0:8001",
                          models_loaded=["gpt2"], status=InferenceStatus.READY,
                          capacity=100, current_load=10),
            InferenceNode(node_id="worker-1", endpoint="http://w1:8001",
                          models_loaded=["gpt2"], status=InferenceStatus.READY,
                          capacity=100, current_load=30),
        ]
        for n in node_configs:
            lb.register_node(n)

        route = lb.route_request(InferenceRequest(model_id="gpt2", prompt="test"))
        assert route == "worker-0"

    @pytest.mark.asyncio
    async def test_pipeline_failover(self):
        lb = InferenceLoadBalancer(RoutingStrategy.LEAST_LOADED)
        n1 = InferenceNode(node_id="healthy", endpoint="http://h:8001",
                          models_loaded=["gpt2"], status=InferenceStatus.READY,
                          capacity=100, current_load=0, health_score=1.0)
        n2 = InferenceNode(node_id="degraded", endpoint="http://d:8001",
                          models_loaded=["gpt2"], status=InferenceStatus.READY,
                          capacity=100, current_load=50, health_score=0.2)
        lb.register_node(n1)
        lb.register_node(n2)

        route = lb.route_request(InferenceRequest(model_id="gpt2", prompt="failover"))
        assert route == "healthy"

    @pytest.mark.asyncio
    async def test_pipeline_distributed_kv_cache(self):
        local = KVCacheManager(max_size_mb=512.0)
        dkc = DistributedKVCache(local, node_id="node-0")

        dkc.register_peer_cache("node-1", {"entries": 100, "size_mb": 50.0})
        dkc.register_peer_cache("node-2", {"entries": 200, "size_mb": 80.0})

        local.put("gpt2", "shared prompt", {"layer0": np.zeros((12, 64))})
        affinity = dkc.compute_cache_affinity("gpt2", "shared prompt")
        assert affinity is not None
        assert len(affinity) > 0

        node_id, cache_id = dkc.find_cached_request("gpt2", "shared prompt")
        assert node_id == "node-0"

        stats = dkc.get_aggregate_stats()
        assert stats["peer_count"] == 2
        assert stats["aggregate_size_mb"] > 50.0


# ============================================================
# Edge cases and error handling
# ============================================================

class TestInferenceEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_prompt(self):
        engine = InferenceEngine(node_id="edge-node")
        await engine.start()
        config = ModelServeConfig(model_id="edge-model")
        await engine.load_model(config)
        req = InferenceRequest(model_id="edge-model", prompt="")
        resp = await engine.infer(req)
        assert resp.error is None
        assert resp.tokens_generated > 0
        await engine.stop()

    @pytest.mark.asyncio
    async def test_max_tokens_boundary(self):
        engine = InferenceEngine(node_id="edge-node")
        await engine.start()
        config = ModelServeConfig(model_id="edge-model")
        await engine.load_model(config)
        req = InferenceRequest(model_id="edge-model", prompt="test", max_tokens=8192)
        resp = await engine.infer(req)
        assert resp.error is None
        await engine.stop()

    @pytest.mark.asyncio
    async def test_zero_temperature(self):
        engine = InferenceEngine(node_id="edge-node")
        await engine.start()
        config = ModelServeConfig(model_id="edge-model")
        await engine.load_model(config)
        req = InferenceRequest(model_id="edge-model", prompt="test", temperature=0.0)
        resp = await engine.infer(req)
        assert resp.error is None
        await engine.stop()

    @pytest.mark.asyncio
    async def test_high_priority_request(self):
        engine = InferenceEngine(node_id="edge-node")
        await engine.start()
        config = ModelServeConfig(model_id="edge-model")
        await engine.load_model(config)
        req = InferenceRequest(model_id="edge-model", prompt="urgent", priority=10)
        resp = await engine.infer(req)
        assert resp.error is None
        await engine.stop()

    @pytest.mark.asyncio
    async def test_inference_timeout(self):
        engine = InferenceEngine(node_id="timeout-node")
        await engine.start()
        config = ModelServeConfig(model_id="slow-model")
        await engine.load_model(config)
        req = InferenceRequest(model_id="slow-model", prompt="test", timeout_ms=1000)
        resp = await engine.infer(req)
        assert resp is not None
        await engine.stop()

    def test_inference_request_validation(self):
        req = InferenceRequest(model_id="test", prompt="hello", max_tokens=100, temperature=0.7)
        assert req.max_tokens == 100
        assert req.temperature == 0.7
        assert 0.0 <= req.top_p <= 1.0
        assert 1 <= req.top_k <= 1000

    def test_model_replica_properties(self):
        rep = ModelReplica(model_id="test", capacity=100, current_load=50,
                          status=InferenceStatus.READY)
        assert rep.is_available is True
        assert rep.load_factor == 0.5
        assert rep.avg_latency == 0.0

    def test_model_replica_at_capacity(self):
        rep = ModelReplica(model_id="test", capacity=100, current_load=100,
                          status=InferenceStatus.READY)
        assert rep.is_available is False

    def test_model_replica_draining(self):
        rep = ModelReplica(model_id="test", capacity=100, current_load=0,
                          status=InferenceStatus.DRAINING)
        assert rep.is_available is False

    def test_metrics_deque_maxlen(self):
        engine = InferenceEngine(node_id="metrics-node")
        assert engine.metrics_history.maxlen == 10000


# ============================================================
# Performance / latency benchmarks
# ============================================================

class TestInferencePerformance:
    @pytest.mark.asyncio
    async def test_single_inference_latency(self):
        engine = InferenceEngine(node_id="perf-node")
        await engine.start()
        config = ModelServeConfig(model_id="perf-model")
        await engine.load_model(config)
        req = InferenceRequest(model_id="perf-model", prompt="Performance test")
        t0 = time.time()
        resp = await engine.infer(req)
        elapsed = (time.time() - t0) * 1000
        assert resp.error is None
        assert elapsed < 5000, f"Inference took {elapsed:.0f}ms (>5s)"
        await engine.stop()

    @pytest.mark.asyncio
    async def test_concurrent_throughput(self):
        engine = InferenceEngine(node_id="throughput-node")
        await engine.start()
        config = ModelServeConfig(model_id="throughput-model")
        await engine.load_model(config)
        num_requests = 50
        tasks = [
            engine.infer(InferenceRequest(model_id="throughput-model", prompt=f"req-{i}"))
            for i in range(num_requests)
        ]
        t0 = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - t0
        errors = [r for r in results if r.error is not None]
        assert len(errors) == 0, f"{len(errors)} errors out of {num_requests}"
        rps = num_requests / elapsed
        assert rps > 10, f"Throughput {rps:.1f} req/s is too low"
        await engine.stop()

    @pytest.mark.asyncio
    async def test_kv_cache_hit_rate(self):
        cache = KVCacheManager(max_size_mb=100.0, ttl_seconds=600.0)
        data = {"layer0": np.random.randn(12, 64).astype(np.float32)}
        for i in range(100):
            cache.put("gpt2", f"prompt-{i}", data)
        for i in range(50):
            cache.get("gpt2", f"prompt-{i}")
        stats = cache.get_stats()
        assert stats["hit_rate"] > 0.4, f"Hit rate {stats['hit_rate']:.2f} too low"

    @pytest.mark.asyncio
    async def test_lb_routing_speed(self):
        lb = InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)
        for i in range(10):
            n = InferenceNode(node_id=f"n{i}", endpoint=f"http://n{i}:8001",
                              models_loaded=["gpt2"], status=InferenceStatus.READY,
                              gpu_count=1, avg_latency_ms=float(i * 10))
            lb.register_node(n)
        t0 = time.time()
        for _ in range(1000):
            lb.route_request(InferenceRequest(model_id="gpt2", prompt="speed test"))
        elapsed = (time.time() - t0) * 1000
        assert elapsed < 500, f"1000 route decisions took {elapsed:.0f}ms"

    @pytest.mark.asyncio
    async def test_autoloader_plan_speed(self):
        catalog = {
            "models": [
                {"model_id": f"m{i}", "name": f"Model {i}", "architecture": "llama",
                 "size_class": "mini" if i < 5 else "small",
                 "params_m": 50 + i * 100,
                 "vram_required_mb": {"q4_k_m": 500 + i * 500}}
                for i in range(20)
            ]
        }
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(catalog, f)
            path = Path(f.name)
        reg = ModelRegistry(catalog_path=path)
        assert reg.load_local() is True
        loader = AutoLoader(reg, available_vram_mb=32000)
        t0 = time.time()
        for _ in range(100):
            loader.compute_load_plan()
        elapsed = (time.time() - t0) * 1000
        assert elapsed < 1000, f"100 load plans took {elapsed:.0f}ms"
        path.unlink()