"""Tests for the native inference engine, pipeline executor, and model downloader."""

import asyncio
import json
import math
import os
import tempfile
import time

import numpy as np
import pytest

from netai.inference.native_engine import (
    NativeInferenceEngine,
    TransformerConfig,
    LayerShard,
    LayerResult,
    _softmax,
    _layer_norm,
    _gelu,
    _silu,
    _rope_positions,
    _apply_rope,
)
from netai.inference.pipeline_executor import (
    PipelineExecutor,
    PipelineStage,
    PipelineResult,
)
from netai.inference.downloader import (
    ModelDownloader,
    HFModelSource,
    DownloadProgress,
    ModelDownload,
    DEFAULT_CACHE_DIR,
)
from netai.inference.engine import InferenceEngine


class TestTransformerConfig:
    def test_defaults(self):
        config = TransformerConfig()
        assert config.vocab_size == 50257
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.num_heads == 12
        assert config.model_type == "gpt2"

    def test_from_hf_config(self):
        hf = {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "model_type": "llama",
            "rope_theta": 500000.0,
        }
        config = TransformerConfig.from_hf_config(hf)
        assert config.vocab_size == 32000
        assert config.hidden_size == 4096
        assert config.num_layers == 32
        assert config.model_type == "llama"
        assert config.rope_theta == 500000.0

    def test_total_params_gpt2(self):
        config = TransformerConfig(hidden_size=768, num_layers=12, num_heads=12,
                                  intermediate_size=3072, vocab_size=50257)
        params = config.total_params()
        assert params > 0
        assert 100e6 < params < 200e6

    def test_total_params_llama(self):
        config = TransformerConfig(hidden_size=4096, num_layers=32, num_heads=32,
                                  intermediate_size=11008, vocab_size=32000)
        params = config.total_params()
        assert params > 6e9

    def test_vram_mb(self):
        config = TransformerConfig()
        vram = config.vram_mb()
        assert vram > 0

    def test_head_dim(self):
        config = TransformerConfig(hidden_size=768, num_heads=12)
        assert config.head_dim() == 64

    def test_vram_per_stage(self):
        config = TransformerConfig(hidden_size=768, num_layers=12, num_heads=12)
        per_stage = config.vram_per_stage(num_stages=4)
        total = config.vram_mb()
        assert per_stage < total
        assert per_stage > 0


class TestMathFunctions:
    def test_softmax(self):
        x = np.array([[1.0, 2.0, 3.0]])
        result = _softmax(x, axis=-1)
        assert result.shape == (1, 3)
        assert abs(result.sum() - 1.0) < 1e-5
        assert result[0, 2] > result[0, 1] > result[0, 0]

    def test_softmax_numerical_stability(self):
        x = np.array([[1e5, 1e5 + 1, 1e5 + 2]])
        result = _softmax(x, axis=-1)
        assert not np.any(np.isnan(result))
        assert abs(result.sum() - 1.0) < 1e-5

    def test_layer_norm(self):
        x = np.random.randn(2, 10).astype(np.float32)
        weight = np.ones(10, dtype=np.float32)
        bias = np.zeros(10, dtype=np.float32)
        result = _layer_norm(x, weight, bias)
        assert result.shape == x.shape
        mean = result.mean(axis=-1)
        assert abs(mean[0]) < 0.1

    def test_gelu(self):
        x = np.array([0.0, 1.0, -1.0, 2.0])
        result = _gelu(x)
        assert result[0] < 0.01
        assert result[1] > 0.5
        assert result[2] < 0
        assert result[1] < x[1]

    def test_silu(self):
        x = np.array([0.0, 1.0, -1.0])
        result = _silu(x)
        expected = x / (1 + np.exp(-x))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_rope_positions(self):
        cos_a, sin_a = _rope_positions(16, 64)
        assert cos_a.shape == (16, 32)
        assert sin_a.shape == (16, 32)
        assert np.all(cos_a >= -1) and np.all(cos_a <= 1)
        assert np.all(sin_a >= -1) and np.all(sin_a <= 1)

    def test_apply_rope(self):
        hidden = np.random.randn(1, 8, 4, 64).astype(np.float32)
        cos_a, sin_a = _rope_positions(8, 64)
        result = _apply_rope(hidden, cos_a, sin_a)
        assert result.shape == hidden.shape
        assert not np.allclose(result, hidden)


class TestNativeInferenceEngine:
    def setup_method(self):
        self.engine = NativeInferenceEngine(node_id="test_node")

    def test_init(self):
        assert self.engine.node_id == "test_node"
        assert len(self.engine.layers) == 0
        assert len(self.engine._loaded_models) == 0

    def test_get_status_empty(self):
        status = self.engine.get_status()
        assert status["node_id"] == "test_node"
        assert status["num_layers_loaded"] == 0
        assert status["weights_memory_mb"] == 0.0
        assert "numpy" in status["backends"]

    def test_load_model_from_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "vocab_size": 100,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 64,
                "max_position_embeddings": 64,
                "model_type": "gpt2",
            }
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)

            rng = np.random.default_rng(42)
            w = rng.standard_normal((100, 32)).astype(np.float32) * 0.02
            qkv = rng.standard_normal((32, 96)).astype(np.float32) * 0.02
            qkv_b = rng.standard_normal((96,)).astype(np.float32) * 0.02
            proj = rng.standard_normal((32, 32)).astype(np.float32) * 0.02
            proj_b = rng.standard_normal((32,)).astype(np.float32) * 0.02
            ffn_w = rng.standard_normal((32, 64)).astype(np.float32) * 0.02
            ffn_b = rng.standard_normal((64,)).astype(np.float32) * 0.02
            ffn_proj = rng.standard_normal((64, 32)).astype(np.float32) * 0.02
            ffn_proj_b = rng.standard_normal((32,)).astype(np.float32) * 0.02
            ln1_w = rng.standard_normal((32,)).astype(np.float32) * 0.1 + 1.0
            ln1_b = np.zeros(32, dtype=np.float32)

            for i in range(2):
                prefix = f"transformer.h.{i}."
                weights = {
                    prefix + "attn.c_attn.weight": qkv,
                    prefix + "attn.c_attn.bias": qkv_b,
                    prefix + "attn.c_proj.weight": proj,
                    prefix + "attn.c_proj.bias": proj_b,
                    prefix + "ln_1.weight": ln1_w.copy(),
                    prefix + "ln_1.bias": ln1_b.copy(),
                    prefix + "mlp.c_fc.weight": ffn_w,
                    prefix + "mlp.c_fc.bias": ffn_b,
                    prefix + "mlp.c_proj.weight": ffn_proj,
                    prefix + "mlp.c_proj.bias": ffn_proj_b,
                    prefix + "ln_2.weight": ln1_w.copy(),
                    prefix + "ln_2.bias": ln1_b.copy(),
                }
                np.savez(os.path.join(tmpdir, f"layer_{i}.npz"), **weights)

            weights = {"wte.weight": w}
            np.savez(os.path.join(tmpdir, "embed.npz"), **weights)

            shard = self.engine.load_model("test-gpt2", tmpdir)
            assert shard.loaded
            assert shard.num_layers == 2
            assert shard.memory_mb > 0
            assert "test-gpt2" in self.engine._loaded_models

    def test_load_nonexistent_model(self):
        shard = self.engine.load_model("nonexistent", "/tmp/no_such_dir_xyz")
        assert not shard.loaded

    def test_forward_layer_without_weights(self):
        config = TransformerConfig(hidden_size=32, num_layers=2, num_heads=2, vocab_size=100)
        self.engine.configs["test-model"] = config
        hidden = np.random.randn(1, 4, 32).astype(np.float32)
        result = self.engine.forward_layer(hidden, "test-model", 0)
        assert result.shape == hidden.shape

    def test_generate_without_embed(self):
        config = TransformerConfig()
        self.engine.configs["no-embed-model"] = config
        result = self.engine.generate("no-embed-model", [0, 1, 2])
        assert "error" in result

    def test_unload_model(self):
        self.engine.configs["unload-test"] = TransformerConfig()
        self.engine._loaded_models.add("unload-test")
        self.engine.layers["unload-test/layer_0"] = {"w": np.zeros((2, 2))}
        result = self.engine.unload_model("unload-test")
        assert result is True
        assert "unload-test" not in self.engine.configs
        assert len(self.engine.layers) == 0

    def test_forward_segment(self):
        config = TransformerConfig(hidden_size=32, num_layers=2, num_heads=2, vocab_size=100)
        self.engine.configs["seg-test"] = config
        hidden = np.random.randn(1, 4, 32).astype(np.float32)
        result = self.engine.forward_segment(hidden, "seg-test", 0, 1)
        assert isinstance(result, LayerResult)
        assert result.latency_ms > 0

    def test_tokenize_simple(self):
        tokens = self.engine._tokenize_simple("hello", vocab_size=50257)
        assert len(tokens) == 5
        assert all(0 <= t < 50257 for t in tokens)

    def test_tokenize_empty(self):
        tokens = self.engine._tokenize_simple("", vocab_size=50257)
        assert tokens == [0]


class TestPipelineExecutor:
    def setup_method(self):
        self.engine = NativeInferenceEngine(node_id="test_node")
        self.executor = PipelineExecutor(local_engine=self.engine)

    def test_plan_pipeline(self):
        config = TransformerConfig(
            hidden_size=256, num_layers=8, num_heads=4,
            intermediate_size=512, vocab_size=1000,
        )
        nodes = [
            {"node_id": "node1", "vram_available_mb": 2048},
            {"node_id": "node2", "vram_available_mb": 1024},
        ]
        stages = self.executor.plan_pipeline("test-model", config, nodes)
        assert len(stages) >= 1
        assert all(isinstance(s, PipelineStage) for s in stages)
        assert stages[0].model_id == "test-model"

    def test_plan_pipeline_empty_nodes(self):
        config = TransformerConfig(num_layers=4)
        stages = self.executor.plan_pipeline("empty", config, [])
        assert len(stages) > 0

    def test_pipeline_assignment(self):
        config = TransformerConfig(
            hidden_size=256, num_layers=12, num_heads=4,
            vocab_size=1000, intermediate_size=512,
        )
        nodes = [
            {"node_id": "test_node", "vram_available_mb": 8192},
        ]
        stages = self.executor.plan_pipeline("assign-test", config, nodes)
        local = self.executor.assign_local_stages("assign-test")
        assert len(local) >= 1

    def test_get_pipeline_status(self):
        config = TransformerConfig(num_layers=4)
        self.executor.plan_pipeline("status-test", config, [{"node_id": "n1", "vram_available_mb": 4096}])
        status = self.executor.get_pipeline_status("status-test")
        assert status["model_id"] == "status-test"
        assert status["stages"] >= 1

    def test_remove_pipeline(self):
        config = TransformerConfig(num_layers=4)
        self.executor.plan_pipeline("rm-test", config, [{"node_id": "n1", "vram_available_mb": 4096}])
        result = self.executor.remove_pipeline("rm-test")
        assert result is True
        assert "rm-test" not in self.executor.pipelines

    def test_list_pipelines(self):
        config = TransformerConfig(num_layers=4)
        self.executor.plan_pipeline("list1", config, [{"node_id": "n1", "vram_available_mb": 4096}])
        self.executor.plan_pipeline("list2", config, [{"node_id": "n2", "vram_available_mb": 4096}])
        result = self.executor.list_pipelines()
        assert len(result) == 2

    def test_pipeline_result_model(self):
        result = PipelineResult(
            request_id="test",
            model_id="test-model",
            generated_tokens=[1, 2, 3],
            total_latency_ms=100.0,
            tokens_per_second=30.0,
            num_stages=2,
        )
        assert result.request_id == "test"
        assert len(result.generated_tokens) == 3

    def test_generate_autoregressive_no_model(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.executor.generate_autoregressive("nonexistent", [0, 1], max_tokens=5)
        )
        assert "error" in result.model_dump()


class TestModelDownloader:
    def setup_method(self):
        self.downloader = ModelDownloader(cache_dir=tempfile.mkdtemp())

    def test_init(self):
        assert self.downloader.cache_dir.startswith(tempfile.gettempdir())
        assert self.downloader.max_concurrent == 4

    def test_mit_allowed_licenses(self):
        for lic in ["mit", "apache-2.0", "bsd-3-clause", "mpl-2.0"]:
            assert lic in ModelDownloader.MIT_ALLOWED

    def test_hf_model_source_url(self):
        source = HFModelSource(model_id="gpt2", revision="main")
        assert source.hub_url == "https://huggingface.co/gpt2"
        assert source.file_url("config.json") == "https://huggingface.co/gpt2/resolve/main/config.json"

    def test_hf_model_source_size(self):
        source = HFModelSource(model_id="test", total_size_bytes=1024 * 1024 * 500)
        assert source.size_mb == 500.0

    def test_list_cached_models_empty(self):
        models = self.downloader.list_cached_models()
        assert models == []

    def test_get_local_model_nonexistent(self):
        result = self.downloader.get_local_model("nonexistent/model")
        assert result is None

    def test_delete_cached_model(self):
        result = self.downloader.delete_cached_model("nonexistent")
        assert result is False

    def test_download_progress(self):
        progress = DownloadProgress(model_id="test", filename="model.bin", total_bytes=1000)
        progress.bytes_downloaded = 500
        assert progress.percent == 50.0
        assert not progress.is_complete

    def test_download_progress_complete(self):
        progress = DownloadProgress(model_id="test", filename="model.bin",
                                    total_bytes=1000, bytes_downloaded=1000)
        progress.finished_at = time.time()
        assert progress.is_complete

    def test_model_download(self):
        dl = ModelDownload(model_id="test-model", files={"a.bin": "/tmp/a.bin"}, total_bytes=1024, verified=True)
        assert dl.size_mb == 1024 / (1024 * 1024)
        assert dl.is_ready

    def test_model_download_not_verified(self):
        dl = ModelDownload(model_id="test-model", files={}, total_bytes=0, verified=False)
        assert not dl.is_ready

    def test_get_download_status_empty(self):
        status = self.downloader.get_download_status()
        assert isinstance(status, dict)
        assert len(status) == 0

    def test_cache_dir_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = os.path.join(tmpdir, "new_cache")
            dl = ModelDownloader(cache_dir=cache)
            assert os.path.isdir(cache)


class TestHFModelSource:
    def test_revision_url(self):
        source = HFModelSource(model_id="facebook/opt-125m", revision="v1.0")
        url = source.file_url("config.json")
        assert "facebook/opt-125m" in url
        assert "v1.0" in url

    def test_sha256_map(self):
        source = HFModelSource(
            model_id="gpt2",
            sha256_map={"model.safetensors": "abc123"},
        )
        assert source.sha256_map["model.safetensors"] == "abc123"


class TestLayerShard:
    def test_creation(self):
        shard = LayerShard(
            model_id="test-model",
            layer_start=0,
            layer_end=11,
            num_layers=12,
            loaded=True,
            memory_mb=500.0,
        )
        assert shard.model_id == "test-model"
        assert shard.num_layers == 12
        assert shard.loaded

    def test_defaults(self):
        shard = LayerShard()
        assert shard.layer_start == 0
        assert not shard.loaded


class TestPipelineStage:
    def test_creation(self):
        stage = PipelineStage(
            node_id="node1",
            model_id="llama-7b",
            stage_index=0,
            total_stages=3,
            layer_start=0,
            layer_end=10,
            num_layers=11,
            vram_mb=2048.0,
        )
        assert stage.node_id == "node1"
        assert stage.num_layers == 11

    def test_avg_latency(self):
        stage = PipelineStage(inference_count=10, total_latency_ms=500.0)
        assert stage.avg_latency_ms == 50.0


class TestInferenceEngineIntegration:
    def test_get_native_engine(self):
        engine = InferenceEngine(node_id="test")
        native = engine.get_native_engine()
        assert isinstance(native, NativeInferenceEngine)
        assert native.node_id == "test"

    def test_get_pipeline_executor(self):
        engine = InferenceEngine(node_id="test")
        executor = engine.get_pipeline_executor()
        assert isinstance(executor, PipelineExecutor)

    def test_get_model_downloader(self):
        engine = InferenceEngine(node_id="test")
        downloader = engine.get_model_downloader()
        assert isinstance(downloader, ModelDownloader)

    def test_status_includes_native(self):
        engine = InferenceEngine(node_id="test")
        status = engine.get_status()
        assert status["native_engine"] is None
        native = engine.get_native_engine()
        status = engine.get_status()
        assert status["native_engine"] is not None

    def test_load_local_model_invalid_dir(self):
        engine = InferenceEngine(node_id="test")
        result = engine.load_local_model("test", "/no/such/dir")
        # Should still work but load 0 layers
        assert "model_id" in result

    def test_native_infer_no_model(self):
        engine = InferenceEngine(node_id="test")
        async def _test():
            result = await engine.native_infer("nonexistent", [0, 1])
            assert "error" in result
        asyncio.get_event_loop().run_until_complete(_test())


class TestDownloadProgress:
    def test_speed_calculation(self):
        progress = DownloadProgress(
            model_id="test", filename="model.bin", total_bytes=10_000_000,
        )
        progress.bytes_downloaded = 5_000_000
        import time
        progress.started_at = time.time() - 1.0
        assert progress.speed_mbps > 0

    def test_zero_total(self):
        progress = DownloadProgress(model_id="test", filename="f.bin")
        assert progress.percent == 0.0


class TestEndToEndNativeGeneration:
    """Test the full generation pipeline with small synthetic model."""

    def _create_small_model(self, tmpdir):
        config = {
            "vocab_size": 50,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "max_position_embeddings": 32,
            "model_type": "gpt2",
        }
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config, f)

        rng = np.random.default_rng(42)
        V, H, I = 50, 16, 32

        weights = {}
        weights["wte.weight"] = rng.standard_normal((V, H)).astype(np.float32) * 0.02

        for i in range(2):
            p = f"transformer.h.{i}."
            weights[p + "attn.c_attn.weight"] = rng.standard_normal((H, 3 * H)).astype(np.float32) * 0.02
            weights[p + "attn.c_attn.bias"] = np.zeros(3 * H, dtype=np.float32)
            weights[p + "attn.c_proj.weight"] = rng.standard_normal((H, H)).astype(np.float32) * 0.02
            weights[p + "attn.c_proj.bias"] = np.zeros(H, dtype=np.float32)
            weights[p + "ln_1.weight"] = np.ones(H, dtype=np.float32)
            weights[p + "ln_1.bias"] = np.zeros(H, dtype=np.float32)
            weights[p + "mlp.c_fc.weight"] = rng.standard_normal((H, I)).astype(np.float32) * 0.02
            weights[p + "mlp.c_fc.bias"] = np.zeros(I, dtype=np.float32)
            weights[p + "mlp.c_proj.weight"] = rng.standard_normal((I, H)).astype(np.float32) * 0.02
            weights[p + "mlp.c_proj.bias"] = np.zeros(H, dtype=np.float32)
            weights[p + "ln_2.weight"] = np.ones(H, dtype=np.float32)
            weights[p + "ln_2.bias"] = np.zeros(H, dtype=np.float32)

        np.savez(os.path.join(tmpdir, "model.npz"), **weights)
        return config

    def test_full_generation_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._create_small_model(tmpdir)

            engine = NativeInferenceEngine(node_id="gen-test")
            shard = engine.load_model("tiny-gpt2", tmpdir)
            assert shard.loaded
            assert shard.num_layers == 2

            embed = engine.embed_tokens.get("tiny-gpt2")
            assert embed is not None
            assert embed.shape[0] == config["vocab_size"]

            result = engine.generate(
                model_id="tiny-gpt2",
                prompt_tokens=[1, 2, 3],
                max_tokens=10,
                temperature=0.8,
            )
            assert "tokens" in result
            assert len(result["tokens"]) == 3 + result["num_generated"]
            assert result["num_generated"] == 10
            assert result["latency_ms"] > 0
            assert result["tokens_per_second"] > 0

            for i in range(result["num_generated"]):
                assert 0 <= result["tokens"][3 + i] < config["vocab_size"]

    def test_pipeline_executor_with_small_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dict = self._create_small_model(tmpdir)

            engine = NativeInferenceEngine(node_id="pipe-test")
            shard = engine.load_model("tiny-gpt2", tmpdir)
            assert shard.loaded

            config = TransformerConfig.from_hf_config(config_dict)
            executor = PipelineExecutor(local_engine=engine)

            nodes = [{"node_id": "pipe-test", "vram_available_mb": 8192}]
            stages = executor.plan_pipeline("tiny-gpt2", config, nodes)
            assert len(stages) >= 1

            result = asyncio.get_event_loop().run_until_complete(
                executor.generate_autoregressive(
                    model_id="tiny-gpt2",
                    prompt_tokens=[1, 2, 3],
                    max_tokens=5,
                    temperature=0.8,
                )
            )
            assert result.model_id == "tiny-gpt2"
            assert len(result.generated_tokens) == 5
            assert result.num_stages >= 1

    def test_forward_layer_preserves_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_small_model(tmpdir)
            engine = NativeInferenceEngine(node_id="shape-test")
            engine.load_model("tiny-gpt2", tmpdir)

            hidden = np.random.randn(1, 8, 16).astype(np.float32)
            output = engine.forward_layer(hidden, "tiny-gpt2", 0)
            assert output.shape == hidden.shape

    def test_pipeline_status_after_plan(self):
        config = TransformerConfig(
            hidden_size=256, num_layers=12, num_heads=4,
            vocab_size=1000, intermediate_size=512,
        )
        executor = PipelineExecutor()
        nodes = [
            {"node_id": "n1", "vram_available_mb": 4096},
            {"node_id": "n2", "vram_available_mb": 2048},
        ]
        stages = executor.plan_pipeline("test", config, nodes)
        status = executor.get_pipeline_status("test")
        assert status["model_id"] == "test"
        assert status["stages"] == len(stages)

    def test_unload_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_small_model(tmpdir)
            engine = NativeInferenceEngine(node_id="reload-test")
            shard = engine.load_model("tiny-gpt2", tmpdir)
            assert shard.loaded

            status = engine.get_status()
            assert "tiny-gpt2" in status["loaded_models"]

            engine.unload_model("tiny-gpt2")
            status = engine.get_status()
            assert "tiny-gpt2" not in status["loaded_models"]

            shard2 = engine.load_model("tiny-gpt2", tmpdir)
            assert shard2.loaded