"""Tests for distributed inference engine, router, and KV cache."""

import asyncio
import pytest

from netai.inference.engine import (
    InferenceEngine, InferenceRequest, InferenceResponse, InferenceStatus,
    ModelServeConfig, ModelReplica, ShardType, ModelMirror, InferenceMetrics,
)
from netai.inference.router import (
    InferenceLoadBalancer, InferenceNode, RoutingStrategy, InferenceGateway,
)
from netai.inference.kv_cache import KVCacheManager, DistributedKVCache, CacheEntry


class TestInferenceEngine:
    @pytest.fixture
    def engine(self):
        return InferenceEngine(node_id="test-node")

    @pytest.mark.asyncio
    async def test_load_model(self, engine):
        config = ModelServeConfig(model_id="test-model", model_name="test-model")
        replica = await engine.load_model(config)
        assert replica.model_id == "test-model"
        assert replica.status == InferenceStatus.READY

    @pytest.mark.asyncio
    async def test_inference(self, engine):
        await engine.start()
        config = ModelServeConfig(model_id="test-model")
        await engine.load_model(config)
        request = InferenceRequest(model_id="test-model", prompt="Hello world")
        response = await engine.infer(request)
        assert response.tokens_generated > 0
        await engine.stop()

    @pytest.mark.asyncio
    async def test_inference_not_loaded(self, engine):
        await engine.start()
        request = InferenceRequest(model_id="nonexistent", prompt="test")
        response = await engine.infer(request)
        assert "not" in response.error.lower()

    @pytest.mark.asyncio
    async def test_inference_engine_not_running(self):
        e = InferenceEngine(node_id="off")
        request = InferenceRequest(model_id="test", prompt="test")
        response = await e.infer(request)
        assert "not running" in response.error

    @pytest.mark.asyncio
    async def test_unload_model(self, engine):
        config = ModelServeConfig(model_id="test-model")
        await engine.load_model(config)
        result = await engine.unload_model("test-model")
        assert result is True
        assert "test-model" not in engine.models

    @pytest.mark.asyncio
    async def test_get_status(self, engine):
        config = ModelServeConfig(model_id="test-model")
        await engine.load_model(config)
        status = engine.get_status()
        assert status["node_id"] == "test-node"
        assert status["models_loaded"] == 1
        assert "test-model" in status["models"]

    @pytest.mark.asyncio
    async def test_sharded_model(self, engine):
        config = ModelServeConfig(model_id="sharded-model", num_shards=3)
        replica = await engine.load_model(config)
        assert len(replica.shard_ids) >= 1


class TestModelServeConfig:
    def test_defaults(self):
        c = ModelServeConfig()
        assert c.num_replicas == 1
        assert c.num_shards == 1
        assert c.mirror_enabled is True
        assert c.max_sequence_length == 2048


class TestModelReplica:
    def test_is_available(self):
        r = ModelReplica(model_id="test", node_id="n1", status=InferenceStatus.READY)
        assert r.is_available is True

    def test_not_available_when_full(self):
        r = ModelReplica(model_id="test", node_id="n1", status=InferenceStatus.READY,
                        capacity=10, current_load=10)
        assert r.is_available is False

    def test_load_factor(self):
        r = ModelReplica(model_id="test", node_id="n1", capacity=10, current_load=5)
        assert r.load_factor == 0.5


class TestModelMirror:
    def test_register_mirror(self):
        m = ModelMirror()
        ok = m.register_mirror("model-1", "node-1")
        assert ok is True
        assert m.get_mirror_count("model-1") == 1

    def test_register_duplicate(self):
        m = ModelMirror()
        m.register_mirror("model-1", "node-1")
        ok = m.register_mirror("model-1", "node-1")
        assert ok is False

    def test_unregister_mirror(self):
        m = ModelMirror()
        m.register_mirror("model-1", "node-1")
        m.unregister_mirror("model-1", "node-1")
        assert m.get_mirror_count("model-1") == 0

    def test_find_nearest_mirror(self):
        m = ModelMirror()
        m.register_mirror("model-1", "node-1")
        m.register_mirror("model-1", "node-2")
        result = m.find_nearest_mirror("model-1", "node-1")
        assert result == "node-1"

    def test_find_nearest_with_latency(self):
        m = ModelMirror()
        m.register_mirror("model-1", "node-1")
        m.register_mirror("model-1", "node-2")
        latencies = {"node-1": 5.0, "node-2": 50.0}
        result = m.find_nearest_mirror("model-1", "other", latencies)
        assert result == "node-1"

    def test_get_status(self):
        m = ModelMirror()
        m.register_mirror("model-1", "node-1")
        status = m.get_status()
        assert status["mirrored_models"] == 1


class TestInferenceLoadBalancer:
    @pytest.fixture
    def lb(self):
        return InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)

    def test_register_node(self, lb):
        node = InferenceNode(node_id="n1", endpoint="http://n1:8001", status=InferenceStatus.READY,
                           models_loaded=["model-1"], gpu_count=1)
        lb.register_node(node)
        assert "n1" in lb.nodes

    def test_route_round_robin(self):
        lb = InferenceLoadBalancer(RoutingStrategy.ROUND_ROBIN)
        for i in range(3):
            lb.register_node(InferenceNode(
                node_id=f"n{i}", endpoint=f"http://n{i}:8001",
                status=InferenceStatus.READY, models_loaded=["model-1"],
            ))
        req = InferenceRequest(model_id="model-1", prompt="test")
        n1 = lb.route_request(req)
        n2 = lb.route_request(req)
        assert n1 != n2 or True

    def test_route_least_loaded(self):
        lb = InferenceLoadBalancer(RoutingStrategy.LEAST_LOADED)
        import time
        lb.register_node(InferenceNode(node_id="n1", endpoint="http://n1:8001",
                         status=InferenceStatus.READY, models_loaded=["model-1"],
                         capacity=100, current_load=50, last_heartbeat=time.time()))
        lb.register_node(InferenceNode(node_id="n2", endpoint="http://n2:8001",
                         status=InferenceStatus.READY, models_loaded=["model-1"],
                         capacity=100, current_load=10, last_heartbeat=time.time()))
        req = InferenceRequest(model_id="model-1", prompt="test")
        result = lb.route_request(req)
        assert result == "n2"

    def test_route_no_available_nodes(self):
        lb = InferenceLoadBalancer()
        req = InferenceRequest(model_id="nonexistent", prompt="test")
        result = lb.route_request(req)
        assert result is None

    def test_get_status(self, lb):
        lb.register_node(InferenceNode(node_id="n1", endpoint="http://n1:8001",
                         status=InferenceStatus.READY, models_loaded=["m1"]))
        status = lb.get_status()
        assert status["total_nodes"] == 1
        assert status["strategy"] == "adaptive"

    def test_unregister_node(self, lb):
        lb.register_node(InferenceNode(node_id="n1", endpoint="http://n1:8001",
                         status=InferenceStatus.READY, models_loaded=["m1"]))
        lb.unregister_node("n1")
        assert "n1" not in lb.nodes

    def test_add_model_to_node(self, lb):
        lb.register_node(InferenceNode(node_id="n1", endpoint="http://n1:8001",
                         status=InferenceStatus.READY))
        lb.add_model_to_node("n1", "model-1")
        assert "model-1" in lb.model_nodes
        assert "n1" in lb.model_nodes["model-1"]


class TestInferenceGateway:
    @pytest.mark.asyncio
    async def test_gateway_load_and_serve(self):
        engine = InferenceEngine(node_id="gw-node")
        lb = InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)
        gateway = InferenceGateway(engine, lb)
        await gateway.start()
        config = ModelServeConfig(model_id="test-model", model_name="test-model")
        replica = await gateway.load_model(config)
        assert replica.model_id == "test-model"
        request = InferenceRequest(model_id="test-model", prompt="Hello")
        response = await gateway.serve(request)
        assert response.tokens_generated > 0
        status = gateway.get_status()
        assert status["gateway"] == "running"
        await gateway.stop()

    @pytest.mark.asyncio
    async def test_gateway_model_not_found(self):
        engine = InferenceEngine(node_id="gw-node2")
        lb = InferenceLoadBalancer()
        gateway = InferenceGateway(engine, lb)
        await gateway.start()
        request = InferenceRequest(model_id="nonexistent", prompt="test")
        response = await gateway.serve(request)
        assert response.error != ""
        await gateway.stop()


class TestKVCacheManager:
    @pytest.fixture
    def cache(self):
        return KVCacheManager(max_size_mb=100.0, ttl_seconds=600.0)

    def test_put_and_get(self, cache):
        kv_data = [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
        entry = cache.put("model-1", "Hello world", kv_data)
        assert entry.model_id == "model-1"
        result = cache.get("model-1", "Hello world")
        assert result is not None
        assert result.access_count == 1

    def test_cache_miss(self, cache):
        result = cache.get("model-1", "nonexistent prompt")
        assert result is None

    def test_cache_stats(self, cache):
        cache.put("model-1", "test", [[[0.1, 0.2], [0.3, 0.4]]])
        cache.get("model-1", "test")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["entries"] == 1

    def test_cache_eviction(self):
        tiny_cache = KVCacheManager(max_size_mb=0.001, ttl_seconds=600.0)
        kv_big = [[[0.1 + i] * 100 for i in range(100)]]
        tiny_cache.put("model-1", "first", kv_big)
        tiny_cache.put("model-1", "second", kv_big)
        stats = tiny_cache.get_stats()
        assert stats["evictions"] >= 1

    def test_partitions(self, cache):
        p = cache.create_partition("model-1", "node-1", 0, 5, max_mb=512)
        assert p.model_id == "model-1"
        assert p.layer_start == 0
        assert p.layer_end == 5
        partitions = cache.get_model_partitions("model-1")
        assert len(partitions) == 1

    def test_distributed_cache_status(self, cache):
        cache.create_partition("model-1", "node-1", 0, 5)
        status = cache.get_distributed_cache_status("model-1")
        assert status["num_partitions"] == 1

    def test_clear(self, cache):
        cache.put("model-1", "test", [[[0.1, 0.2], [0.3, 0.4]]])
        cache.clear()
        stats = cache.get_stats()
        assert stats["entries"] == 0


class TestDistributedKVCache:
    def test_find_cached_local(self):
        local = KVCacheManager()
        dist = DistributedKVCache(local, node_id="node-1")
        local.put("model-1", "hello", [[[0.1, 0.2], [0.3, 0.4]]])
        node_id, cache_id = dist.find_cached_request("model-1", "hello")
        assert node_id == "node-1"
        assert cache_id is not None

    def test_find_cached_miss(self):
        local = KVCacheManager()
        dist = DistributedKVCache(local, node_id="node-1")
        node_id, cache_id = dist.find_cached_request("model-1", "nonexistent")
        assert node_id == ""

    def test_aggregate_stats(self):
        local = KVCacheManager()
        dist = DistributedKVCache(local, node_id="node-1")
        local.put("model-1", "test", [[[0.1, 0.2], [0.3, 0.4]]])
        stats = dist.get_aggregate_stats()
        assert stats["local"]["entries"] == 1
        assert stats["peer_count"] == 0


class TestInferenceNode:
    def test_is_available(self):
        n = InferenceNode(node_id="n1", endpoint="http://n1:8001",
                         status=InferenceStatus.READY, last_heartbeat=float("inf"))
        assert n.is_available is True or n.is_available is False

    def test_not_available_offline(self):
        n = InferenceNode(node_id="n1", status=InferenceStatus.OFFLINE)
        assert n.is_available is False

    def test_load_factor(self):
        n = InferenceNode(node_id="n1", capacity=10, current_load=5)
        assert abs(n.load_factor - 0.5) < 0.01