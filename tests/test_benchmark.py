"""Tests for benchmark runner module."""

from __future__ import annotations

import json
import os
import tempfile
import time

import numpy as np
import pytest
from pydantic import BaseModel

from netai.benchmark.runner import (
    ModelBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
    InferenceMetrics,
    MemoryMetrics,
    StartupMetrics,
    PipelineMetrics,
    TimingResult,
)
from netai.inference.native_engine import NativeInferenceEngine, TransformerConfig


@pytest.fixture
def mock_engine():
    engine = NativeInferenceEngine(node_id="bench-test-node")
    config = TransformerConfig(
        vocab_size=1024,
        hidden_size=128,
        num_layers=4,
        num_heads=8,
        intermediate_size=512,
        max_position_embeddings=1024,
        layer_norm_eps=1e-5,
        model_type="gpt2",
    )
    engine.configs["bench-model"] = config
    embed = np.random.randn(1024, 128).astype(np.float32) * 0.02
    engine.embed_tokens["bench-model"] = embed
    engine._loaded_models.add("bench-model")

    for i in range(config.num_layers):
        layer_weights = {}
        layer_weights["ln_1.weight"] = np.ones(128, dtype=np.float32)
        layer_weights["ln_1.bias"] = np.zeros(128, dtype=np.float32)
        layer_weights["ln_2.weight"] = np.ones(128, dtype=np.float32)
        layer_weights["ln_2.bias"] = np.zeros(128, dtype=np.float32)
        layer_weights["attn.c_attn.weight"] = np.random.randn(128, 3 * 128).astype(np.float32) * 0.02
        layer_weights["attn.c_attn.bias"] = np.zeros(3 * 128, dtype=np.float32)
        layer_weights["attn.c_proj.weight"] = np.random.randn(128, 128).astype(np.float32) * 0.02
        layer_weights["attn.c_proj.bias"] = np.zeros(128, dtype=np.float32)
        layer_weights["mlp.c_fc.weight"] = np.random.randn(128, 512).astype(np.float32) * 0.02
        layer_weights["mlp.c_fc.bias"] = np.zeros(512, dtype=np.float32)
        layer_weights["mlp.c_proj.weight"] = np.random.randn(512, 128).astype(np.float32) * 0.02
        layer_weights["mlp.c_proj.bias"] = np.zeros(128, dtype=np.float32)
        engine.layers[f"bench-model/layer_{i}"] = layer_weights

    engine.layer_norm_f["bench-model"] = (
        np.ones(128, dtype=np.float32),
        np.zeros(128, dtype=np.float32),
    )

    return engine


@pytest.fixture
def benchmark_runner(mock_engine):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "models", "bench-model")
        os.makedirs(model_dir, exist_ok=True)
        config_json = {
            "vocab_size": 1024,
            "hidden_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "intermediate_size": 512,
            "model_type": "gpt2",
        }
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config_json, f)
        yield ModelBenchmark(engine=mock_engine, cache_dir=tmpdir)


@pytest.fixture
def mock_runner():
    return ModelBenchmark(engine=None)


class TestBenchmarkConfig:
    def test_default_config(self):
        cfg = BenchmarkConfig()
        assert cfg.model_id == ""
        assert cfg.warmup_iterations == 3
        assert cfg.benchmark_iterations == 10
        assert cfg.max_tokens == 64
        assert cfg.batch_sizes == [1, 2, 4, 8]
        assert len(cfg.prompt_token_counts) == 5

    def test_custom_config(self):
        cfg = BenchmarkConfig(
            model_id="my-model",
            warmup_iterations=1,
            benchmark_iterations=5,
            max_tokens=128,
            batch_sizes=[1, 2],
            seed=99,
        )
        assert cfg.model_id == "my-model"
        assert cfg.max_tokens == 128
        assert cfg.batch_sizes == [1, 2]
        assert cfg.seed == 99

    def test_config_serialization(self):
        cfg = BenchmarkConfig(model_id="test-model", max_tokens=32)
        d = cfg.model_dump()
        assert d["model_id"] == "test-model"
        assert d["max_tokens"] == 32

    def test_config_deserialization(self):
        d = {"model_id": "deserial", "max_tokens": 16, "batch_sizes": [1, 4]}
        cfg = BenchmarkConfig(**d)
        assert cfg.model_id == "deserial"
        assert cfg.max_tokens == 16
        assert cfg.batch_sizes == [1, 4]


class TestBenchmarkResult:
    def test_empty_result(self):
        result = BenchmarkResult(model_id="empty")
        d = result.model_dump()
        assert d["model_id"] == "empty"
        assert d["error"] == ""
        assert isinstance(d["inference"], list)
        assert isinstance(d["memory"], list)

    def test_result_with_error(self):
        result = BenchmarkResult(model_id="bad-model", error="Model not found")
        assert result.error == "Model not found"


class TestInferenceMetrics:
    def test_default_metrics(self):
        m = InferenceMetrics()
        assert m.prompt_tokens == 0
        assert m.generated_tokens == 0
        assert m.tokens_per_second == 0.0

    def test_valid_metrics(self):
        timing = TimingResult(mean_ms=100.0, p95_ms=120.0, samples=10)
        m = InferenceMetrics(
            prompt_tokens=8,
            generated_tokens=64,
            total_latency=timing,
            tokens_per_second=640.0,
            time_to_first_token_ms=30.0,
            inter_token_latency_ms=1.1,
        )
        assert m.prompt_tokens == 8
        assert m.generated_tokens == 64
        assert m.total_latency.mean_ms == 100.0
        assert m.total_latency.p95_ms == 120.0
        assert m.tokens_per_second == 640.0


class TestTimingResult:
    def test_compute_timing_from_samples(self):
        from netai.benchmark.runner import _compute_timing
        result = _compute_timing([10.0, 12.0, 8.0, 15.0, 11.0])
        assert result.samples == 5
        assert 8.0 <= result.mean_ms <= 15.0
        assert result.min_ms == 8.0
        assert result.max_ms == 15.0

    def test_empty_timing(self):
        from netai.benchmark.runner import _compute_timing
        result = _compute_timing([])
        assert result.samples == 0


class TestModelBenchmarkMock:
    """Tests with no engine (mock mode)."""

    def test_mock_inference_returns_metrics(self, mock_runner):
        metrics = mock_runner.benchmark_inference("test-model", prompt_tokens=8, max_tokens=16)
        assert isinstance(metrics, InferenceMetrics)
        assert metrics.prompt_tokens == 8
        assert metrics.generated_tokens == 16
        assert metrics.tokens_per_second > 0
        assert metrics.total_latency.samples > 0

    def test_mock_memory_all_batch_sizes(self, mock_runner):
        results = mock_runner.benchmark_memory("test-model", batch_sizes=[1, 2, 4, 8])
        assert len(results) == 4
        for r in results:
            assert isinstance(r, MemoryMetrics)
            assert r.weights_mb > 0
            assert r.batch_size > 0

    def test_mock_startup(self, mock_runner):
        metrics = mock_runner.benchmark_startup("test-model")
        assert isinstance(metrics, StartupMetrics)
        assert metrics.load_time_s > 0
        assert metrics.weights_mb > 0
        assert metrics.layers_loaded > 0

    def test_mock_pipeline(self, mock_runner):
        metrics = mock_runner.benchmark_pipeline("test-model", num_stages=4)
        assert isinstance(metrics, PipelineMetrics)
        assert metrics.num_stages == 4
        assert metrics.total_overhead_ms > 0
        assert 0 <= metrics.efficiency <= 100

    def test_mock_run_suite(self, mock_runner):
        result = mock_runner.run_suite("mock-model")
        assert isinstance(result, BenchmarkResult)
        assert result.model_id == "mock-model"
        assert len(result.inference) == 5
        assert len(result.memory) == 4
        assert result.startup is not None
        assert result.pipeline is not None
        assert result.summary
        assert result.summary.get("max_tokens_per_second", 0) > 0
        assert result.error == ""

    def test_mock_generate_report(self, mock_runner):
        mock_runner.run_suite("report-model")
        report = mock_runner.generate_report("report-model")
        assert "## Model: `report-model`" in report
        assert "Inference Performance" in report
        assert "Memory Usage" in report
        assert "Startup" in report
        assert "Pipeline" in report

    def test_mock_compare_models(self, mock_runner):
        comparison = mock_runner.compare_models(["model-a", "model-b"])
        assert "models" in comparison
        assert "model-a" in comparison["models"]
        assert "model-b" in comparison["models"]
        assert comparison["models"]["model-a"]["tokens_per_second"] > 0
        assert comparison["models"]["model-a"]["memory_weights_mb"] > 0

    def test_mock_report_no_results(self, mock_runner):
        from netai.benchmark.runner import ModelBenchmark
        fresh = ModelBenchmark(engine=None)
        report = fresh.generate_report()
        assert "No benchmark results" in report


class TestModelBenchmarkReal:
    """Tests with a loaded engine."""

    def test_inference_with_engine(self, benchmark_runner):
        metrics = benchmark_runner.benchmark_inference("bench-model", prompt_tokens=4, max_tokens=8)
        assert metrics.prompt_tokens == 4
        assert metrics.generated_tokens == 8
        assert metrics.tokens_per_second > 0
        assert metrics.total_latency.mean_ms > 0

    def test_inference_different_prompt_lengths(self, benchmark_runner):
        for pt in [1, 8, 32]:
            metrics = benchmark_runner.benchmark_inference("bench-model", prompt_tokens=pt, max_tokens=16)
            assert metrics.prompt_tokens == pt

    def test_memory_with_engine(self, benchmark_runner):
        results = benchmark_runner.benchmark_memory("bench-model", batch_sizes=[1, 2])
        assert len(results) == 2
        for r in results:
            assert r.weights_mb > 0
            assert r.batch_size in (1, 2)

    def test_startup_with_engine(self, benchmark_runner):
        metrics = benchmark_runner.benchmark_startup("bench-model")
        assert metrics.load_time_s >= 0
        assert metrics.layers_loaded >= 0

    def test_pipeline_with_engine(self, benchmark_runner):
        metrics = benchmark_runner.benchmark_pipeline("bench-model", num_stages=2)
        assert metrics.num_stages == 2
        assert metrics.total_overhead_ms > 0
        assert metrics.compute_time_ms > 0

    def test_full_suite_with_engine(self, benchmark_runner):
        result = benchmark_runner.run_suite("bench-model")
        assert result.model_id == "bench-model"
        assert len(result.inference) > 0
        assert len(result.memory) > 0
        assert result.summary["total_weights_mb"] > 0
        assert result.startup.layers_loaded >= 0
        assert result.error == ""

    def test_unknown_model_returns_error(self, mock_runner):
        result = mock_runner.run_suite("nonexistent")
        assert result.model_id == "nonexistent"
        assert isinstance(result, BenchmarkResult)

    def test_results_persist(self, benchmark_runner):
        benchmark_runner.run_suite("bench-model")
        all_results = benchmark_runner.results
        assert "bench-model" in all_results
        assert isinstance(all_results["bench-model"], BenchmarkResult)


class TestBenchmarkConcurrency:
    def test_thread_safe_run_suite(self, mock_runner):
        import threading

        errors = []

        def run_suite():
            try:
                mock_runner.run_suite("thread-model")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=run_suite) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert "thread-model" in mock_runner.results


class TestBenchmarkEdgeCases:
    def test_zero_prompt_tokens(self, mock_runner):
        metrics = mock_runner.benchmark_inference("test", prompt_tokens=0, max_tokens=10)
        assert isinstance(metrics, InferenceMetrics)

    def test_single_token_generation(self, mock_runner):
        metrics = mock_runner.benchmark_inference("test", prompt_tokens=1, max_tokens=1)
        assert metrics.generated_tokens == 1
        assert metrics.tokens_per_second > 0

    def test_large_max_tokens(self, mock_runner):
        result = mock_runner.run_suite("large-model", BenchmarkConfig(model_id="large-model", max_tokens=512))
        assert result.error == ""
        assert result.inference[0].generated_tokens == 512

    def test_empty_compare_models(self, mock_runner):
        comparison = mock_runner.compare_models([])
        assert "models" in comparison
        assert comparison["models"] == {}

    def test_config_bounds(self):
        with pytest.raises(Exception):
            BenchmarkConfig(warmup_iterations=200)

        with pytest.raises(Exception):
            BenchmarkConfig(max_tokens=5000)
