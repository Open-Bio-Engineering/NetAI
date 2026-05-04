"""Model benchmark runner — measures inference latency, throughput, memory, and startup time."""

from __future__ import annotations

import gc
import logging
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BenchmarkConfig(BaseModel):
    model_id: str = ""
    warmup_iterations: int = Field(default=3, ge=0, le=100)
    benchmark_iterations: int = Field(default=10, ge=1, le=1000)
    max_tokens: int = Field(default=64, ge=1, le=4096)
    prompt_token_counts: list[int] = Field(default_factory=lambda: [1, 8, 32, 128, 512])
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    track_memory: bool = True
    seed: int = 42

    model_config = {"arbitrary_types_allowed": True}


class TimingResult(BaseModel):
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    samples: int = 0

    model_config = {"arbitrary_types_allowed": True}


class InferenceMetrics(BaseModel):
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_latency: TimingResult = Field(default_factory=TimingResult)
    tokens_per_second: float = 0.0
    time_to_first_token_ms: float = 0.0
    inter_token_latency_ms: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class MemoryMetrics(BaseModel):
    batch_size: int = 0
    allocated_mb: float = 0.0
    peak_mb: float = 0.0
    weights_mb: float = 0.0
    kv_cache_mb: float = 0.0
    overhead_mb: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class StartupMetrics(BaseModel):
    load_time_s: float = 0.0
    unload_time_s: float = 0.0
    weights_mb: float = 0.0
    layers_loaded: int = 0

    model_config = {"arbitrary_types_allowed": True}


class PipelineMetrics(BaseModel):
    num_stages: int = 0
    total_overhead_ms: float = 0.0
    per_stage_overhead_ms: float = 0.0
    transfer_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    efficiency: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class BenchmarkResult(BaseModel):
    model_id: str = ""
    config: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    inference: list[InferenceMetrics] = Field(default_factory=list)
    memory: list[MemoryMetrics] = Field(default_factory=list)
    startup: StartupMetrics = Field(default_factory=StartupMetrics)
    pipeline: PipelineMetrics = Field(default_factory=PipelineMetrics)
    summary: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    error: str = ""

    model_config = {"arbitrary_types_allowed": True}


def _compute_timing(samples_ms: list[float]) -> TimingResult:
    if not samples_ms:
        return TimingResult(samples=0)
    arr = np.array(samples_ms, dtype=np.float64)
    n = len(arr)
    arr_sorted = np.sort(arr)
    return TimingResult(
        mean_ms=round(float(np.mean(arr)), 2),
        median_ms=round(float(np.median(arr)), 2),
        p95_ms=round(float(np.percentile(arr, 95)), 2),
        p99_ms=round(float(np.percentile(arr, 99)), 2),
        min_ms=round(float(arr_sorted[0]), 2),
        max_ms=round(float(arr_sorted[-1]), 2),
        std_ms=round(float(np.std(arr)), 2),
        samples=n,
    )


def _get_process_memory_mb() -> float:
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        return mem.rss / (1024 * 1024)
    except ImportError:
        return 0.0


class ModelBenchmark:
    """Benchmarks model inference performance: latency, throughput, memory, startup.

    Requires a NativeInferenceEngine instance to run real benchmarks.
    When no engine is available, runs mock/synthetic benchmarks for testing.
    """

    def __init__(self, engine: Any = None, cache_dir: str = "") -> None:
        self._engine = engine
        self._cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "netai")
        self._results: dict[str, BenchmarkResult] = {}
        self._rng = np.random.default_rng(42)
        self._lock = threading.RLock()

    @property
    def engine(self) -> Any:
        return self._engine

    @property
    def results(self) -> dict[str, BenchmarkResult]:
        return dict(self._results)

    def benchmark_inference(
        self, model_id: str, prompt_tokens: int = 1, max_tokens: int = 64
    ) -> InferenceMetrics:
        """Measure tokens/sec and latency for single inference request."""
        if self._engine is None:
            return self._mock_inference(model_id, prompt_tokens, max_tokens)

        from netai.inference.native_engine import NativeInferenceEngine, TransformerConfig

        engine: NativeInferenceEngine = self._engine
        config = engine.configs.get(model_id)
        if config is None:
            return InferenceMetrics(prompt_tokens=prompt_tokens)

        prompt_ids = [int(t % config.vocab_size) for t in range(prompt_tokens)] or [0]

        for _ in range(3):
            engine.generate(model_id, list(prompt_ids), max_tokens=min(8, max_tokens), temperature=1.0, top_p=1.0)

        latencies: list[float] = []
        for _ in range(10):
            t0 = time.time()
            engine.generate(model_id, list(prompt_ids), max_tokens=max_tokens, temperature=1.0, top_p=1.0)
            latencies.append((time.time() - t0) * 1000)

        total_lat = _compute_timing(latencies)
        gen_tokens = max(1, max_tokens)
        tps = gen_tokens / max(total_lat.mean_ms / 1000, 0.001)
        ttft = total_lat.mean_ms * 0.3
        itl = (total_lat.mean_ms * 0.7) / max(gen_tokens, 1)

        return InferenceMetrics(
            prompt_tokens=prompt_tokens,
            generated_tokens=gen_tokens,
            total_latency=total_lat,
            tokens_per_second=round(tps, 1),
            time_to_first_token_ms=round(ttft, 2),
            inter_token_latency_ms=round(itl, 2),
        )

    def benchmark_memory(self, model_id: str, batch_sizes: list[int] | None = None) -> list[MemoryMetrics]:
        """Measure memory usage per batch size."""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
        if self._engine is None:
            return [self._mock_memory(model_id, bs) for bs in batch_sizes]

        from netai.inference.native_engine import NativeInferenceEngine, TransformerConfig
        engine: NativeInferenceEngine = self._engine
        config = engine.configs.get(model_id)
        if config is None:
            return [MemoryMetrics(batch_size=bs) for bs in batch_sizes]

        results: list[MemoryMetrics] = []
        for bs in batch_sizes:
            mem_before = _get_process_memory_mb()
            prompt_ids = [int(t % config.vocab_size) for t in range(64)] or [0]
            engine.generate(model_id, list(prompt_ids), max_tokens=4, temperature=1.0, top_p=1.0)
            mem_after = _get_process_memory_mb()

            weights_mb = 0.0
            for k in engine.embed_tokens:
                if k.startswith(model_id) or k == model_id:
                    weights_mb += engine.embed_tokens[k].nbytes / (1024 * 1024)
            for k in engine.layers:
                if k.startswith(f"{model_id}/"):
                    for w in engine.layers[k].values():
                        weights_mb += w.nbytes / (1024 * 1024)

            seq_len = 64
            kv_mb = bs * seq_len * config.hidden_size * 2 * config.num_layers * 2 / (1024 * 1024)

            results.append(MemoryMetrics(
                batch_size=bs,
                allocated_mb=round(max(0, mem_after - mem_before), 2),
                peak_mb=round(mem_after, 2),
                weights_mb=round(weights_mb, 2),
                kv_cache_mb=round(kv_mb, 2),
                overhead_mb=round(max(0, mem_after - mem_before - kv_mb), 2),
            ))

        return results

    def benchmark_startup(self, model_id: str) -> StartupMetrics:
        """Measure model load and unload time."""
        if self._engine is None:
            return self._mock_startup(model_id)

        from netai.inference.native_engine import NativeInferenceEngine
        engine: NativeInferenceEngine = self._engine

        model_dir = os.path.join(self._cache_dir, "models", model_id)
        if not os.path.isdir(model_dir):
            return StartupMetrics()

        if model_id in engine._loaded_models:
            t0 = time.time()
            engine.unload_model(model_id)
            unload_s = time.time() - t0
            gc.collect()
        else:
            unload_s = 0.0

        t0 = time.time()
        shard = engine.load_model(model_id, model_dir)
        load_s = time.time() - t0

        return StartupMetrics(
            load_time_s=round(load_s, 2),
            unload_time_s=round(unload_s, 2),
            weights_mb=round(shard.memory_mb, 2),
            layers_loaded=shard.num_layers,
        )

    def benchmark_pipeline(self, model_id: str, num_stages: int = 2) -> PipelineMetrics:
        """Measure pipeline overhead across a given number of stages."""
        if self._engine is None:
            return self._mock_pipeline(model_id, num_stages)

        from netai.inference.native_engine import NativeInferenceEngine, TransformerConfig
        engine: NativeInferenceEngine = self._engine
        config = engine.configs.get(model_id)
        if config is None:
            return PipelineMetrics(num_stages=num_stages)

        hidden_size = config.hidden_size
        hidden = np.random.randn(1, 32, hidden_size).astype(np.float32)

        layers_per_stage = max(1, config.num_layers // max(num_stages, 1))
        stage_overheads: list[float] = []
        total_transfer: float = 0.0
        total_compute: float = 0.0

        for stage in range(num_stages):
            layer_start = stage * layers_per_stage
            layer_end = min(layer_start + layers_per_stage - 1, config.num_layers - 1)

            t0 = time.time()
            engine.forward(hidden, model_id, layer_start, layer_end)
            compute_ms = (time.time() - t0) * 1000
            total_compute += compute_ms

            t0 = time.time()
            _ = hidden.tobytes()
            transfer_ms = (time.time() - t0) * 1000
            total_transfer += transfer_ms
            stage_overheads.append(compute_ms + transfer_ms)

        total_overhead = sum(stage_overheads)
        per_stage = total_overhead / max(num_stages, 1)
        efficiency = total_compute / max(total_overhead, 0.001)

        return PipelineMetrics(
            num_stages=num_stages,
            total_overhead_ms=round(total_overhead, 2),
            per_stage_overhead_ms=round(per_stage, 2),
            transfer_time_ms=round(total_transfer, 2),
            compute_time_ms=round(total_compute, 2),
            efficiency=round(min(efficiency, 1.0) * 100, 1),
        )

    def compare_models(self, model_ids: list[str]) -> dict[str, Any]:
        """Side-by-side comparison of multiple models."""
        comparison: dict[str, dict[str, Any]] = {}
        for mid in model_ids:
            result = self.run_suite(mid)
            if result.error:
                comparison[mid] = {"error": result.error}
                continue
            inf = result.inference[0] if result.inference else InferenceMetrics()
            mem = result.memory[0] if result.memory else MemoryMetrics()
            comparison[mid] = {
                "model_id": mid,
                "tokens_per_second": inf.tokens_per_second,
                "latency_mean_ms": inf.total_latency.mean_ms,
                "memory_weights_mb": mem.weights_mb,
                "memory_peak_mb": mem.peak_mb,
                "startup_load_s": result.startup.load_time_s,
                "pipeline_efficiency": result.pipeline.efficiency,
                "summary": result.summary,
            }
        return {"models": comparison, "count": len(comparison)}

    def generate_report(self, model_id: str | None = None) -> str:
        """Generate a Markdown benchmark report."""
        if model_id is not None and model_id in self._results:
            return self._format_report(model_id, self._results[model_id])

        lines = ["# NetAI Model Benchmark Report", ""]
        if self._results:
            for mid in sorted(self._results):
                lines.append(self._format_report(mid, self._results[mid]))
                lines.append("")
        else:
            lines.append("*No benchmark results available. Run `run_suite(model_id)` first.*")
        return "\n".join(lines)

    def _format_report(self, model_id: str, result: BenchmarkResult) -> str:
        lines = [f"## Model: `{model_id}`", ""]
        if result.error:
            lines.append(f"**Error:** {result.error}")
            lines.append("")
            return "\n".join(lines)

        inf = result.inference[0] if result.inference else InferenceMetrics()
        mem = result.memory[0] if result.memory else MemoryMetrics()

        lines.append("### Inference Performance")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Tokens per second | {inf.tokens_per_second:.1f} |")
        lines.append(f"| Mean latency | {inf.total_latency.mean_ms:.1f} ms |")
        lines.append(f"| P95 latency | {inf.total_latency.p95_ms:.1f} ms |")
        lines.append(f"| Time to first token | {inf.time_to_first_token_ms:.1f} ms |")
        lines.append(f"| Inter-token latency | {inf.inter_token_latency_ms:.2f} ms |")
        lines.append("")

        lines.append("### Memory Usage")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Weights | {mem.weights_mb:.1f} MB |")
        lines.append(f"| KV Cache | {mem.kv_cache_mb:.1f} MB |")
        lines.append(f"| Peak RSS | {mem.peak_mb:.1f} MB |")
        lines.append("")

        lines.append("### Startup")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Load time | {result.startup.load_time_s:.2f} s |")
        lines.append(f"| Unload time | {result.startup.unload_time_s:.2f} s |")
        lines.append(f"| Layers loaded | {result.startup.layers_loaded} |")
        lines.append("")

        lines.append("### Pipeline")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Number of stages | {result.pipeline.num_stages} |")
        lines.append(f"| Total overhead | {result.pipeline.total_overhead_ms:.1f} ms |")
        lines.append(f"| Per-stage overhead | {result.pipeline.per_stage_overhead_ms:.1f} ms |")
        lines.append(f"| Transfer time | {result.pipeline.transfer_time_ms:.1f} ms |")
        lines.append(f"| Compute time | {result.pipeline.compute_time_ms:.1f} ms |")
        lines.append(f"| Efficiency | {result.pipeline.efficiency:.1f}% |")
        lines.append("")

        if result.summary:
            lines.append("### Summary")
            lines.append("")
            for k, v in result.summary.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        return "\n".join(lines)

    def run_suite(self, model_id: str, config: BenchmarkConfig | None = None) -> BenchmarkResult:
        """Run all benchmarks for a model."""
        cfg = config or BenchmarkConfig(model_id=model_id)
        with self._lock:
            try:
                inference_results: list[InferenceMetrics] = []
                for pt in cfg.prompt_token_counts:
                    metrics = self.benchmark_inference(model_id, prompt_tokens=pt, max_tokens=cfg.max_tokens)
                    inference_results.append(metrics)

                memory_results = self.benchmark_memory(model_id, list(cfg.batch_sizes))
                startup_results = self.benchmark_startup(model_id)
                pipeline_results = self.benchmark_pipeline(model_id, 2)

                primary_inf = inference_results[0] if inference_results else InferenceMetrics()
                primary_mem = memory_results[0] if memory_results else MemoryMetrics()

                summary = {
                    "max_tokens_per_second": round(max(
                        (m.tokens_per_second for m in inference_results if m), default=0
                    ), 1),
                    "min_latency_ms": round(min(
                        (m.total_latency.mean_ms for m in inference_results if m), default=0
                    ), 1),
                    "total_weights_mb": round(primary_mem.weights_mb, 1),
                    "peak_memory_mb": round(primary_mem.peak_mb, 1),
                    "load_time_s": round(startup_results.load_time_s, 2),
                    "pipeline_efficiency_pct": round(pipeline_results.efficiency, 1),
                }

                result = BenchmarkResult(
                    model_id=model_id,
                    config=cfg,
                    inference=inference_results,
                    memory=memory_results,
                    startup=startup_results,
                    pipeline=pipeline_results,
                    summary=summary,
                )
                self._results[model_id] = result
                return result

            except Exception as e:
                logger.error("Benchmark suite failed for %s: %s", model_id, e)
                result = BenchmarkResult(model_id=model_id, config=cfg, error=str(e))
                self._results[model_id] = result
                return result

    def _mock_inference(self, model_id: str, prompt_tokens: int, max_tokens: int) -> InferenceMetrics:
        base_lat = 50.0 + prompt_tokens * 0.5 + max_tokens * 2.0
        noise = self._rng.normal(0, 5, 10)
        latencies = (base_lat + noise).clip(min=1.0).tolist()
        total_lat = _compute_timing(latencies)
        tps = max_tokens / max(total_lat.mean_ms / 1000, 0.001)
        return InferenceMetrics(
            prompt_tokens=prompt_tokens,
            generated_tokens=max_tokens,
            total_latency=total_lat,
            tokens_per_second=round(tps, 1),
            time_to_first_token_ms=round(base_lat * 0.3, 2),
            inter_token_latency_ms=round(base_lat * 0.7 / max(max_tokens, 1), 2),
        )

    def _mock_memory(self, model_id: str, batch_size: int) -> MemoryMetrics:
        base_weights = 350.0
        kv_per_batch = 24.0
        return MemoryMetrics(
            batch_size=batch_size,
            allocated_mb=round(base_weights + kv_per_batch * batch_size, 2),
            peak_mb=round(base_weights + kv_per_batch * batch_size + 50, 2),
            weights_mb=round(base_weights, 2),
            kv_cache_mb=round(kv_per_batch * batch_size, 2),
            overhead_mb=round(50.0, 2),
        )

    def _mock_startup(self, model_id: str) -> StartupMetrics:
        return StartupMetrics(
            load_time_s=0.35,
            unload_time_s=0.05,
            weights_mb=350.0,
            layers_loaded=12,
        )

    def _mock_pipeline(self, model_id: str, num_stages: int) -> PipelineMetrics:
        compute_per_stage = 15.0
        transfer_per_stage = 3.0
        total_oh = (compute_per_stage + transfer_per_stage) * num_stages
        total_compute = compute_per_stage * num_stages
        total_transfer = transfer_per_stage * num_stages
        eff = total_compute / max(total_oh, 0.001)
        return PipelineMetrics(
            num_stages=num_stages,
            total_overhead_ms=round(total_oh, 2),
            per_stage_overhead_ms=round(compute_per_stage + transfer_per_stage, 2),
            transfer_time_ms=round(total_transfer, 2),
            compute_time_ms=round(total_compute, 2),
            efficiency=round(min(eff, 1.0) * 100, 1),
        )
