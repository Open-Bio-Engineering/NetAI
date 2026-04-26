"""Tests for resource profiler."""

import pytest
from netai.resource.profiler import ResourceProfiler, ResourceProfile, can_run_model, suggest_batch_size


class TestResourceProfiler:
    def test_profile(self):
        profiler = ResourceProfiler()
        profile = profiler.profile()
        assert isinstance(profile, ResourceProfile)
        assert profile.cpu_cores > 0
        assert profile.ram_total_gb > 0
        assert profile.os_name

    def test_profile_has_python(self):
        profiler = ResourceProfiler()
        profile = profiler.profile()
        assert profile.python_version
        assert "." in profile.python_version


class TestResourceProfile:
    def test_summary(self):
        p = ResourceProfile(cpu_cores=8, cpu_available=4, ram_total_gb=32.0, ram_available_gb=16.0)
        s = p.summary
        assert "CPU:4/8" in s
        assert "RAM:16.0/32.0GB" in s

    def test_summary_with_gpu(self):
        p = ResourceProfile(cpu_cores=8, cpu_available=4, gpu_count=2, gpu_available=1,
                           ram_total_gb=32.0, ram_available_gb=16.0)
        s = p.summary
        assert "GPU:1/2" in s

    def test_training_capacity_score(self):
        p = ResourceProfile(cpu_available=8, gpu_available=2, ram_available_gb=32.0,
                           gpu_available_vram_mb=[8000, 8000])
        score = p.training_capacity_score
        assert score > 0

    def test_total_flops_estimate(self):
        p = ResourceProfile(cpu_cores=8, cpu_freq_ghz=3.0, gpu_vram_mb=[12000])
        flops = p.total_flops_estimate
        assert flops > 0


class TestCanRunModel:
    def test_can_run_gpu(self):
        p = ResourceProfile(gpu_count=1, gpu_available=1, gpu_vram_mb=[24000], ram_available_gb=64)
        ok, device = can_run_model(20.0, p)
        assert ok
        assert device == "gpu"

    def test_can_run_cpu(self):
        p = ResourceProfile(gpu_count=0, ram_available_gb=64)
        ok, device = can_run_model(10.0, p)
        assert ok
        assert device == "cpu"

    def test_cannot_run(self):
        p = ResourceProfile(gpu_count=0, ram_available_gb=4)
        ok, device = can_run_model(10.0, p)
        assert not ok


class TestSuggestBatchSize:
    def test_gpu_batch(self):
        p = ResourceProfile(gpu_available=1, gpu_available_vram_mb=[24000], ram_available_gb=64)
        bs = suggest_batch_size(p, 2.0)
        assert bs >= 1

    def test_cpu_batch(self):
        p = ResourceProfile(gpu_available=0, ram_available_gb=32)
        bs = suggest_batch_size(p, 2.0)
        assert bs >= 1

    def test_minimal_resources(self):
        p = ResourceProfile(gpu_available=0, ram_available_gb=8, gpu_available_vram_mb=[])
        bs = suggest_batch_size(p, 1.0)
        assert bs >= 1