"""Resource profiler - detect CPU, GPU, RAM capabilities."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger(__name__)


@dataclass
class ResourceProfile:
    cpu_cores: int = 0
    cpu_available: int = 0
    cpu_arch: str = ""
    cpu_freq_ghz: float = 0.0
    gpu_count: int = 0
    gpu_available: int = 0
    gpu_names: list[str] = field(default_factory=list)
    gpu_vram_mb: list[int] = field(default_factory=list)
    gpu_available_vram_mb: list[int] = field(default_factory=list)
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    disk_total_gb: float = 0.0
    disk_available_gb: float = 0.0
    os_name: str = ""
    python_version: str = ""
    has_cuda: bool = False
    has_rocm: bool = False
    has_vulkan: bool = False
    torch_available: bool = False
    torch_gpu_count: int = 0

    @property
    def total_flops_estimate(self) -> float:
        cpu_flops = self.cpu_cores * self.cpu_freq_ghz * 8.0
        gpu_flops = 0.0
        for vram in self.gpu_vram_mb:
            if vram > 8000:
                gpu_flops += 30.0
            elif vram > 4000:
                gpu_flops += 15.0
            else:
                gpu_flops += 5.0
        return cpu_flops + gpu_flops

    @property
    def training_capacity_score(self) -> float:
        score = self.cpu_available * 1.0
        score += self.gpu_available * 10.0
        score += min(self.ram_available_gb, 32.0) * 0.5
        for vram in self.gpu_available_vram_mb:
            score += vram / 1024.0 * 2.0
        return score

    @property
    def summary(self) -> str:
        parts = [f"CPU:{self.cpu_available}/{self.cpu_cores}"]
        if self.gpu_count > 0:
            parts.append(f"GPU:{self.gpu_available}/{self.gpu_count}")
        parts.append(f"RAM:{self.ram_available_gb:.1f}/{self.ram_total_gb:.1f}GB")
        return " ".join(parts)


class ResourceProfiler:
    def profile(self) -> ResourceProfile:
        p = ResourceProfile()
        p.cpu_cores = psutil.cpu_count(logical=True) or 0
        p.cpu_available = psutil.cpu_count(logical=False) or p.cpu_cores
        p.cpu_arch = platform.machine()
        try:
            freq = psutil.cpu_freq()
            if freq:
                p.cpu_freq_ghz = freq.current / 1000.0
        except Exception:
            pass
        mem = psutil.virtual_memory()
        p.ram_total_gb = mem.total / (1024**3)
        p.ram_available_gb = mem.available / (1024**3)
        try:
            disk = psutil.disk_usage("/")
            p.disk_total_gb = disk.total / (1024**3)
            p.disk_available_gb = disk.free / (1024**3)
        except Exception:
            pass
        p.os_name = f"{platform.system()} {platform.release()}"
        p.python_version = platform.python_version()
        self._detect_gpus(p)
        self._detect_gpu_backends(p)
        self._detect_torch(p)
        return p

    def _detect_gpus(self, p: ResourceProfile):
        nvidia_gpus = self._detect_nvidia()
        amd_gpus = self._detect_amd()
        all_gpus = nvidia_gpus + amd_gpus
        p.gpu_count = len(all_gpus)
        p.gpu_available = len(all_gpus)
        p.gpu_names = [g[0] for g in all_gpus]
        p.gpu_vram_mb = [g[1] for g in all_gpus]
        p.gpu_available_vram_mb = [g[2] for g in all_gpus]

    def _detect_nvidia(self) -> list[tuple[str, int, int]]:
        gpus = []
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = [x.strip() for x in line.split(",")]
                        if len(parts) >= 3:
                            name = parts[0]
                            total_mb = int(float(parts[1]))
                            free_mb = int(float(parts[2]))
                            gpus.append((name, total_mb, free_mb))
        except Exception:
            pass
        return gpus

    def _detect_amd(self) -> list[tuple[str, int, int]]:
        gpus = []
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                name = "AMD GPU"
                total = 0
                free = 0
                for line in lines:
                    if "Card" in line and "series" in line.lower():
                        name = line.split(":")[-1].strip()
                    elif "Total VRAM" in line:
                        total = int("".join(c for c in line.split(":")[-1] if c.isdigit()) or "0")
                    elif "Total VRAM Heap" in line or "VRAM used" in line.lower():
                        used = int("".join(c for c in line.split(":")[-1] if c.isdigit()) or "0")
                        if total > 0:
                            free = max(0, total - used)
                if total > 0:
                    gpus.append((name, total, free))
        except Exception:
            pass
        try:
            if os.path.exists("/sys/class/drm"):
                for entry in os.listdir("/sys/class/drm"):
                    if entry.startswith("card") and "-" not in entry:
                        gpu_path = f"/sys/class/drm/{entry}/device/uevent"
                        if os.path.exists(gpu_path):
                            gpus.append((f"AMD GPU ({entry})", 0, 0))
                            break
        except Exception:
            pass
        return gpus

    def _detect_gpu_backends(self, p: ResourceProfile):
        try:
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
            p.has_cuda = result.returncode == 0
        except Exception:
            pass
        try:
            result = subprocess.run(["rocm-smi"], capture_output=True, text=True, timeout=5)
            p.has_rocm = result.returncode == 0
        except Exception:
            pass
        try:
            result = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=5)
            p.has_vulkan = result.returncode == 0
        except Exception:
            pass

    def _detect_torch(self, p: ResourceProfile):
        try:
            import torch
            p.torch_available = True
            p.torch_gpu_count = torch.cuda.device_count()
            if p.torch_gpu_count > 0 and p.gpu_count == 0:
                p.gpu_count = p.torch_gpu_count
                p.gpu_available = p.torch_gpu_count
                p.gpu_names = [torch.cuda.get_device_name(i) for i in range(p.torch_gpu_count)]
                p.gpu_vram_mb = [
                    int(torch.cuda.get_device_properties(i).total_mem / 1024**2)
                    for i in range(p.torch_gpu_count)
                ]
                p.gpu_available_vram_mb = [
                    int(torch.cuda.mem_get_info(i)[0] / 1024**2) if i < p.torch_gpu_count else 0
                    for i in range(p.torch_gpu_count)
                ]
                p.has_cuda = True
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                p.gpu_count = max(p.gpu_count, 1)
                p.gpu_available = max(p.gpu_available, 1)
                if "Apple" not in str(p.gpu_names):
                    p.gpu_names.append("Apple Metal Performance Shaders")
        except ImportError:
            pass


def can_run_model(model_params_gb: float, profile: ResourceProfile) -> tuple[bool, str]:
    if profile.gpu_count > 0 and max(profile.gpu_vram_mb, default=0) >= model_params_gb * 1024:
        return True, "gpu"
    if profile.ram_available_gb >= model_params_gb * 1.5:
        return True, "cpu"
    return False, "insufficient"


def suggest_batch_size(profile: ResourceProfile, model_params_gb: float) -> int:
    avail_vram = sum(profile.gpu_available_vram_mb) if profile.gpu_available_vram_mb else 0
    avail_ram = profile.ram_available_gb * 1024
    if profile.gpu_available > 0 and avail_vram > 0:
        overhead = max(model_params_gb * 1024 * 4, 512)
        usable = max(avail_vram - overhead, 0)
        per_sample = max(model_params_gb * 1024 * 2, 256)
        return max(1, int(usable / per_sample))
    if avail_ram > 0:
        overhead = max(model_params_gb * 1024 * 2, 1024)
        usable = max(avail_ram - overhead, 0)
        per_sample = max(model_params_gb * 1024 * 4, 512)
        return max(1, int(usable / per_sample))
    return 1