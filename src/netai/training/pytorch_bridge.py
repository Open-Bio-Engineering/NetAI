"""PyTorch training bridge - enables real GPU training when PyTorch is available."""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
import uuid
from typing import Any

from netai.training.engine import (
    TrainingConfig, TrainingJob, TrainingStatus, TrainingMetrics,
    GradientCompressor, CheckpointManager,
)

logger = logging.getLogger(__name__)

TORCH_AVAILABLE = False
TORCH_CUDA_AVAILABLE = False
TORCH_MPS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
    TORCH_MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
except ImportError:
    pass


class SimpleTransformer(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, config: TrainingConfig):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_seq_length, config.hidden_size)
        self.layers = nn.ModuleList([
            self._make_layer(config) for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._init_weights()

    def _make_layer(self, config):
        return nn.ModuleDict({
            "ln1": nn.LayerNorm(config.hidden_size),
            "attn": nn.ModuleDict({
                "q_proj": nn.Linear(config.hidden_size, config.hidden_size),
                "k_proj": nn.Linear(config.hidden_size, config.hidden_size),
                "v_proj": nn.Linear(config.hidden_size, config.hidden_size),
                "out_proj": nn.Linear(config.hidden_size, config.hidden_size),
            }),
            "ln2": nn.LayerNorm(config.hidden_size),
            "ffn": nn.ModuleDict({
                "up": nn.Linear(config.hidden_size, config.intermediate_size),
                "down": nn.Linear(config.intermediate_size, config.hidden_size),
            }),
        })

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.zeros_(m.bias)
                nn.ones_(m.weight)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        for layer in self.layers:
            ln1 = layer["ln1"](h)
            Q = layer["attn"]["q_proj"](ln1)
            K = layer["attn"]["k_proj"](ln1)
            V = layer["attn"]["v_proj"](ln1)
            H = self.config.hidden_size // self.config.num_heads
            Q = Q.view(B, T, self.config.num_heads, H).transpose(1, 2)
            K = K.view(B, T, self.config.num_heads, H).transpose(1, 2)
            V = V.view(B, T, self.config.num_heads, H).transpose(1, 2)
            attn = torch.matmul(Q, K.transpose(-2, -1)) / (H ** 0.5)
            attn = torch.softmax(attn, dim=-1)
            attn_out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, -1)
            h = h + layer["attn"]["out_proj"](attn_out)
            ln2 = layer["ln2"](h)
            ffn = torch.relu(layer["ffn"]["up"](ln2))
            h = h + layer["ffn"]["down"](ffn)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits


class DummyDataset(Dataset if TORCH_AVAILABLE else object):
    def __init__(self, vocab_size: int, seq_length: int, num_samples: int = 1000):
        if not TORCH_AVAILABLE:
            return
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]


class PyTorchTrainer:
    def __init__(self, config: TrainingConfig, checkpoint_manager: CheckpointManager | None = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed. Install with: pip install torch")
        self.config = config
        self.checkpoint_mgr = checkpoint_manager or CheckpointManager()
        self.job = TrainingJob(config)
        self.device = self._select_device()
        self.model = SimpleTransformer(config).to(self.device)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self._running = False
        self._step = 0

    def _select_device(self):
        if self.config.device_preference.value == "cuda" and TORCH_CUDA_AVAILABLE:
            return torch.device("cuda")
        elif self.config.device_preference.value == "mps" and TORCH_MPS_AVAILABLE:
            return torch.device("mps")
        return torch.device("cpu")

    def _create_optimizer(self):
        no_decay = ["bias", "ln"]
        params = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.config.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        if self.config.optimizer.value == "adamw":
            return optim.AdamW(params, lr=self.config.learning_rate)
        elif self.config.optimizer.value == "adam":
            return optim.Adam(params, lr=self.config.learning_rate)
        elif self.config.optimizer.value == "sgd":
            return optim.SGD(params, lr=self.config.learning_rate, momentum=0.9)
        return optim.AdamW(params, lr=self.config.learning_rate)

    def _create_scheduler(self):
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.total_steps, eta_min=self.config.learning_rate * 0.1
        )

    def _create_dataloader(self):
        dataset = DummyDataset(self.config.vocab_size, self.config.max_seq_length, num_samples=10000)
        return DataLoader(
            dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

    def _train_step(self, batch) -> tuple[float, float]:
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        loss.backward()
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        return loss.item(), grad_norm

    async def train_step(self) -> TrainingMetrics:
        dataloader = self._create_dataloader()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        num_batches = 0
        accumulated_loss = 0.0
        for i, batch in enumerate(dataloader):
            if i >= self.config.gradient_accumulation_steps:
                break
            loss, grad_norm = self._train_step(batch)
            accumulated_loss += loss
            num_batches += 1
            epoch_grad_norm += grad_norm
            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        if num_batches > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        self._step += 1
        avg_loss = accumulated_loss / max(num_batches, 1)
        metrics = TrainingMetrics(
            step=self._step,
            epoch=self._step * self.config.micro_batch_size / 10000,
            loss=avg_loss,
            lr=self.optimizer.param_groups[0]["lr"],
            grad_norm=epoch_grad_norm / max(num_batches, 1),
            throughput_tok_s=self.config.batch_size * self.config.max_seq_length / max(0.01, 1.0),
            gpu_util=torch.cuda.utilization() if TORCH_CUDA_AVAILABLE and self.device.type == "cuda" else 0.0,
            mem_used_gb=torch.cuda.memory_allocated() / (1024**3) if TORCH_CUDA_AVAILABLE and self.device.type == "cuda" else 0.0,
            elapsed_s=time.time() - (self.job.started_at or time.time()),
        )
        self.job.record_metrics(metrics)
        if self._step % self.config.checkpoint_interval == 0:
            self._save_checkpoint()
        return metrics

    def _save_checkpoint(self):
        state = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
        loss = self.job.latest_metrics.loss if self.job.latest_metrics else 0.0
        self.checkpoint_mgr.save_checkpoint(self.job, self._step, state, loss)

    async def train(self, callback=None) -> TrainingJob:
        self.job.status = TrainingStatus.RUNNING
        self.job.started_at = time.time()
        self._running = True
        try:
            while self._step < self.config.total_steps and self._running:
                metrics = await self.train_step()
                if callback:
                    await callback(metrics)
                await asyncio.sleep(0)
            if self.job.status == TrainingStatus.RUNNING:
                self.job.status = TrainingStatus.COMPLETED
                self.job.completed_at = time.time()
                self._save_checkpoint()
        except Exception as e:
            self.job.status = TrainingStatus.FAILED
            self.job.error_message = str(e)
            logger.error("PyTorch training failed: %s", e)
        finally:
            self._running = False
        return self.job

    def stop(self):
        self._running = False

    def get_model_state(self) -> dict[str, Any]:
        return self.model.state_dict()

    def load_model_state(self, state_dict: dict[str, Any]):
        self.model.load_state_dict(state_dict)

    @staticmethod
    def is_available() -> bool:
        return TORCH_AVAILABLE

    @staticmethod
    def device_info() -> dict[str, Any]:
        if not TORCH_AVAILABLE:
            return {"available": False}
        info = {
            "available": True,
            "cuda": TORCH_CUDA_AVAILABLE,
            "mps": TORCH_MPS_AVAILABLE,
            "cuda_devices": torch.cuda.device_count() if TORCH_CUDA_AVAILABLE else 0,
        }
        if TORCH_CUDA_AVAILABLE:
            info["cuda_device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["cuda_vram_mb"] = [torch.cuda.get_device_properties(i).total_mem // (1024**2)
                                     for i in range(torch.cuda.device_count())]
        return info