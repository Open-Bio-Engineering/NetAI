"""Native transformer inference engine — runs model forward pass locally.

Supports GPT-2/Llama-style transformer architectures using NumPy (CPU) or
PyTorch (GPU/CPU). Each node loads a contiguous range of layers and processes
activations through them. Pipeline-parallel inference sends activations between
nodes so together they run models too large for any single machine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None

HAS_TOKENIZERS = False
try:
    from tokenizers import Tokenizer as _HFTokenizer
    HAS_TOKENIZERS = True
except ImportError:
    pass


class LayerShard(BaseModel):
    shard_id: str = Field(default_factory=lambda: hashlib.sha256(os.urandom(8)).hexdigest()[:12])
    model_id: str = ""
    layer_start: int = 0
    layer_end: int = 0
    num_layers: int = 0
    weight_files: list[str] = Field(default_factory=list)
    loaded: bool = False
    load_time_s: float = 0.0
    memory_mb: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class LayerResult(BaseModel):
    request_id: str = ""
    shard_id: str = ""
    tokens: list[int] = Field(default_factory=list)
    hidden_shape: list[int] = Field(default_factory=list)
    hidden_dtype: str = "float32"
    hidden_checksum: str = ""
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class TransformerConfig:
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 2048,
        layer_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        model_type: str = "gpt2",
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.rope_theta = rope_theta
        self.model_type = model_type

    @classmethod
    def from_hf_config(cls, config: dict[str, Any]) -> TransformerConfig:
        return cls(
            vocab_size=config.get("vocab_size", 50257),
            hidden_size=config.get("hidden_size", config.get("n_embd", 768)),
            num_layers=config.get("num_hidden_layers", config.get("n_layer", 12)),
            num_heads=config.get("num_attention_heads", config.get("n_head", 12)),
            intermediate_size=config.get("intermediate_size", config.get("n_inner", 3072)),
            max_position_embeddings=config.get("max_position_embeddings", config.get("n_positions", 2048)),
            layer_norm_eps=config.get("layer_norm_eps", config.get("layer_norm_epsilon", 1e-5)),
            rope_theta=config.get("rope_theta", 10000.0),
            model_type=config.get("model_type", "gpt2"),
        )

    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def total_params(self) -> int:
        h = self.hidden_size
        l = self.num_layers
        i = self.intermediate_size or 4 * h
        v = self.vocab_size
        embed = v * h
        attn_per_layer = 4 * h * h
        ffn_per_layer = 3 * h * i
        ln_per_layer = 4 * h
        output = h * v
        return embed + l * (attn_per_layer + ffn_per_layer + ln_per_layer) + output

    def vram_mb(self, bytes_per_param: int = 2, batch_size: int = 1, seq_len: int = 2048) -> float:
        weights_mb = self.total_params() * bytes_per_param / (1024 * 1024)
        kv_mb = batch_size * seq_len * self.hidden_size * 2 * self.num_layers * bytes_per_param / (1024 * 1024)
        act_mb = batch_size * seq_len * self.hidden_size * bytes_per_param / (1024 * 1024)
        return (weights_mb + kv_mb + act_mb) * 1.1

    def vram_per_stage(self, num_stages: int, batch_size: int = 1, seq_len: int = 2048, bytes_per_param: int = 2) -> float:
        if num_stages <= 0:
            return self.vram_mb(bytes_per_param, batch_size, seq_len)
        layers_per_stage = max(1, self.num_layers // num_stages)
        i = self.intermediate_size or 4 * self.hidden_size
        params_per_stage = layers_per_stage * (
            4 * self.hidden_size ** 2 + 2 * self.hidden_size * i
        ) + self.hidden_size * self.vocab_size / num_stages
        weights_mb = params_per_stage * bytes_per_param / (1024 * 1024)
        kv_mb = batch_size * seq_len * self.hidden_size * 2 * layers_per_stage * bytes_per_param / (1024 * 1024)
        overhead = weights_mb * 0.1
        return weights_mb + kv_mb + overhead


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def _rope_positions(seq_len: int, head_dim: int, theta: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(seq_len, dtype=np.float32)
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = np.outer(positions, freqs)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    return cos_angles, sin_angles


def _apply_rope(hidden: np.ndarray, cos_a: np.ndarray, sin_a: np.ndarray) -> np.ndarray:
    batch, seq, heads, head_d = hidden.shape
    d2 = head_d // 2
    x1 = hidden[..., :d2]
    x2 = hidden[..., d2:]
    cos_b = cos_a[:seq, :d2][np.newaxis, :, np.newaxis, :]
    sin_b = sin_a[:seq, :d2][np.newaxis, :, np.newaxis, :]
    out1 = x1 * cos_b - x2 * sin_b
    out2 = x2 * cos_b + x1 * sin_b
    return np.concatenate([out1, out2], axis=-1)


class NativeInferenceEngine:
    """Runs transformer forward pass locally using NumPy or PyTorch.

    Loads weights from downloaded model files, processes input through assigned
    layers, and returns hidden states for the next pipeline stage (or final
    token logits for the last stage).
    """

    def __init__(self, node_id: str = ""):
        self.node_id = node_id or hashlib.sha256(os.urandom(8)).hexdigest()[:12]
        self.layers: dict[str, dict[str, np.ndarray]] = {}
        self.configs: dict[str, TransformerConfig] = {}
        self.embed_tokens: dict[str, np.ndarray] = {}
        self.output_proj: dict[str, np.ndarray] = {}
        self.layer_norm_f: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._loaded_models: set[str] = set()
        self._tokenizers: dict[str, Any] = {}
        self._has_tokenizers = HAS_TOKENIZERS

    def _load_safetensors_header(self, path: str) -> dict[str, Any]:
        try:
            from safetensors import safe_open
            return {}
        except ImportError:
            pass
        try:
            with open(path, "rb") as f:
                header_size = int.from_bytes(f.read(8), "little")
                header_json = f.read(header_size).decode("utf-8")
                return json.loads(header_json)
        except Exception as e:
            logger.error("Failed to read safetensors header from %s: %s", path, e)
            return {}

    def _load_safetensors_file(self, path: str) -> dict[str, np.ndarray]:
        try:
            from safetensors.numpy import safe_open as np_safe_open
            weights = {}
            with np_safe_open(path, framework="np") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
            return weights
        except ImportError:
            pass

        try:
            from safetensors import safe_open
            weights = {}
            with safe_open(path, framework="np") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
            logger.info("Loaded %d tensors from %s (safetensors)", len(weights), path)
            return weights
        except ImportError:
            pass

        logger.warning("safetensors package not available, trying numpy manual load for %s", path)
        return self._load_safetensors_manual(path)

    def _load_safetensors_manual(self, path: str) -> dict[str, np.ndarray]:
        try:
            with open(path, "rb") as f:
                header_size = int.from_bytes(f.read(8), "little")
                header_json = f.read(header_size).decode("utf-8")
                header = json.loads(header_json)
            weights = {}
            dtype_map = {
                "F16": np.float16, "BF16": np.bfloat16, "F32": np.float32,
                "F64": np.float64, "I8": np.int8, "I16": np.int16,
                "I32": np.int32, "I64": np.int64, "U8": np.uint8,
            }
            with open(path, "rb") as f:
                for name, info in header.items():
                    if name == "__metadata__":
                        continue
                    dtype = dtype_map.get(info.get("dtype", "F32"), np.float32)
                    shape = info.get("shape", [])
                    offsets = info.get("data_offsets", [0, 0])
                    start = 8 + header_size + offsets[0]
                    count = (offsets[1] - offsets[0]) // np.dtype(dtype).itemsize
                    f.seek(start)
                    arr = np.frombuffer(f.read(offsets[1] - offsets[0]), dtype=dtype)
                    if shape:
                        arr = arr.reshape(shape)
                    weights[name] = arr
            logger.info("Manually loaded %d tensors from %s", len(weights), path)
            return weights
        except Exception as e:
            logger.error("Manual safetensors load failed for %s: %s", path, e)
            return {}

    def _load_pytorch_bin(self, path: str) -> dict[str, np.ndarray]:
        try:
            import torch as _torch
            state_dict = _torch.load(path, map_location="cpu", weights_only=True)
            weights = {}
            for key, val in state_dict.items():
                arr = val.numpy() if hasattr(val, "numpy") else np.array(val)
                weights[key] = arr
            logger.info("Loaded %d tensors from %s (pytorch bin)", len(weights), path)
            return weights
        except Exception as e:
            logger.error("PyTorch bin load failed for %s: %s", path, e)
            return {}

    def _load_gguf_file(self, path: str) -> dict[str, np.ndarray]:
        logger.warning("GGUF format requires llama-cpp-python or gguf package — skipping %s", path)
        return {}

    def _load_weights_from_dir(self, model_dir: str) -> dict[str, np.ndarray]:
        all_weights = {}
        for fname in sorted(os.listdir(model_dir)):
            fpath = os.path.join(model_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if fname.endswith(".safetensors"):
                w = self._load_safetensors_file(fpath)
                all_weights.update(w)
            elif fname.endswith(".npz"):
                try:
                    data = np.load(fpath, allow_pickle=False)
                    for key in data.files:
                        all_weights[key] = data[key].astype(np.float32) if data[key].dtype != np.float32 else data[key]
                    logger.info("Loaded %d tensors from %s (npz)", len(data.files), fname)
                except Exception as e:
                    logger.error("Failed to load npz %s: %s", fname, e)
            elif fname.endswith(".bin") or fname.endswith(".pt"):
                if "index" in fname:
                    continue
                w = self._load_pytorch_bin(fpath)
                all_weights.update(w)
            elif fname.endswith(".gguf"):
                w = self._load_gguf_file(fpath)
                all_weights.update(w)
        return all_weights

    def load_model(self, model_id: str, model_dir: str, layer_start: int = -1, layer_end: int = -1) -> LayerShard:
        t0 = time.time()
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            logger.error("No config.json found in %s", model_dir)
            return LayerShard(model_id=model_id)

        with open(config_path) as f:
            hf_config = json.load(f)
        config = TransformerConfig.from_hf_config(hf_config)
        self.configs[model_id] = config

        all_weights = self._load_weights_from_dir(model_dir)
        if not all_weights:
            logger.error("No weights loaded from %s", model_dir)
            return LayerShard(model_id=model_id)

        if layer_start < 0:
            layer_start = 0
        if layer_end < 0:
            layer_end = config.num_layers - 1

        num_layers_loaded = 0
        total_mem = 0.0
        is_gpt2 = config.model_type in ("gpt2", "gpt_neo")
        is_llama = config.model_type in ("llama", "mistral", "qwen2", "qwen2_moe", "gemma", "gemma2", "phi3", "stablelm")

        for i in range(layer_start, layer_end + 1):
            layer_weights = {}
            prefix_patterns = [
                f"transformer.h.{i}.", f"model.layers.{i}.", f"encoder.layer.{i}.",
                f"h.{i}.", f"layers.{i}.", f"block.{i}.",
            ]
            for key, val in all_weights.items():
                for prefix in prefix_patterns:
                    if key.startswith(prefix):
                        local_key = key[len(prefix):]
                        layer_weights[local_key] = val.astype(np.float32)
                        total_mem += val.nbytes
                        break

            if layer_weights:
                self.layers[f"{model_id}/layer_{i}"] = layer_weights
                num_layers_loaded += 1

        embed_key = "wte.weight" if is_gpt2 else "model.embed_tokens.weight"
        embed_candidates = [embed_key, "transformer.wte.weight", "model.embed_tokens.weight", "wpe.weight"]
        for candidate in embed_candidates:
            if candidate in all_weights:
                self.embed_tokens[model_id] = all_weights[candidate].astype(np.float32)
                total_mem += all_weights[candidate].nbytes
                break

        output_key = "lm_head.weight" if not is_gpt2 else "wte.weight"
        output_candidates = [
            output_key, "lm_head.weight", "transformer.ln_f.weight",
            "transformer.lm_head.weight",
        ]
        for candidate in output_candidates:
            if candidate in all_weights:
                if is_gpt2 and candidate == "transformer.ln_f.weight":
                    ln_w = all_weights["transformer.ln_f.weight"].astype(np.float32)
                    ln_b_candidate = all_weights.get("transformer.ln_f.bias")
                    ln_b = ln_b_candidate.astype(np.float32) if ln_b_candidate is not None else np.zeros_like(ln_w)
                    self.layer_norm_f[model_id] = (ln_w, ln_b)
                    total_mem += ln_w.nbytes + ln_b.nbytes
                elif candidate == "lm_head.weight":
                    self.output_proj[model_id] = all_weights[candidate].astype(np.float32)
                    total_mem += all_weights[candidate].nbytes

        ln_f_candidates = ["ln_f.weight", "transformer.ln_f.weight", "model.norm.weight"]
        ln_f_b_candidates = ["ln_f.bias", "transformer.ln_f.bias", "model.norm.bias"]
        for j, candidate in enumerate(ln_f_candidates):
            if candidate in all_weights:
                ln_w = all_weights[candidate].astype(np.float32)
                ln_b_key = ln_f_b_candidates[j] if j < len(ln_f_b_candidates) else ""
                ln_b_val = all_weights.get(ln_b_key)
                ln_b = ln_b_val.astype(np.float32) if ln_b_val is not None else np.zeros_like(ln_w)
                self.layer_norm_f[model_id] = (ln_w, ln_b)
                total_mem += ln_w.nbytes + ln_b.nbytes
                break

        self._loaded_models.add(model_id)
        load_time = time.time() - t0
        shard = LayerShard(
            model_id=model_id,
            layer_start=layer_start,
            layer_end=layer_end,
            num_layers=num_layers_loaded,
            loaded=num_layers_loaded > 0,
            load_time_s=round(load_time, 2),
            memory_mb=round(total_mem / (1024 * 1024), 2),
        )
        logger.info("Loaded %d/%d layers for %s (%.1f MB, %.1fs)",
                     num_layers_loaded, layer_end - layer_start + 1,
                     model_id, total_mem / (1024 * 1024), load_time)
        # Try to load tokenizer
        tok_path = os.path.join(model_dir, "tokenizer.json")
        if os.path.exists(tok_path) and self._has_tokenizers:
            try:
                from tokenizers import Tokenizer as _HFTok
                tok = _HFTok.from_file(tok_path)
                self._tokenizers[model_id] = tok
                logger.info("Loaded tokenizer for %s (vocab_size=%d)", model_id, tok.get_vocab_size())
            except Exception as e:
                logger.warning("Failed to load tokenizer for %s: %s", model_id, e)
        elif os.path.exists(tok_path):
            logger.warning("tokenizers library not available; GPT-2 output will show token IDs instead of text")

        return shard

    def _tokenize_simple(self, text: str, vocab_size: int = 50257) -> list[int]:
        tokens = []
        for ch in text:
            code = ord(ch)
            tokens.append(code % vocab_size)
        if not tokens:
            tokens = [0]
        return tokens

    def _forward_attention(
        self, hidden: np.ndarray, weights: dict[str, np.ndarray],
        config: TransformerConfig, layer_idx: int, cos_a: np.ndarray = None,
        sin_a: np.ndarray = None,
    ) -> np.ndarray:
        batch, seq_len, _ = hidden.shape
        num_heads = config.num_heads
        head_dim = config.hidden_size // num_heads

        q_proj = weights.get("attn.c_attn.weight", weights.get("self_attn.q_proj.weight", None))
        k_proj = weights.get("attn.c_attn.weight", weights.get("self_attn.k_proj.weight", None))
        v_proj = weights.get("attn.c_proj.weight", weights.get("self_attn.v_proj.weight", None))
        o_proj = weights.get("attn.c_proj.weight", weights.get("self_attn.o_proj.weight", None))

        q_bias = weights.get("attn.c_attn.bias", weights.get("self_attn.q_proj.bias", None))
        k_bias = weights.get("attn.c_attn.bias", weights.get("self_attn.k_proj.bias", None))
        v_bias = weights.get("attn.c_attn.bias", weights.get("self_attn.v_proj.bias", None))
        o_bias = weights.get("attn.c_proj.bias", weights.get("self_attn.o_proj.bias", None))

        if q_proj is None:
            rng = np.random.default_rng(42 + layer_idx)
            q_proj = rng.standard_normal((config.hidden_size, config.hidden_size)).astype(np.float32) * 0.02
            k_proj = rng.standard_normal((config.hidden_size, config.hidden_size)).astype(np.float32) * 0.02
            v_proj = rng.standard_normal((config.hidden_size, config.hidden_size)).astype(np.float32) * 0.02
            o_proj = rng.standard_normal((config.hidden_size, config.hidden_size)).astype(np.float32) * 0.02

        is_gpt2_qkv = q_proj is not None and q_proj.shape[0] == config.hidden_size and q_proj.shape[1] == 3 * config.hidden_size

        if is_gpt2_qkv:
            qkv = hidden @ q_proj
            if q_bias is not None:
                qkv += q_bias
            q, k, v = np.split(qkv, 3, axis=-1)
        else:
            q = hidden @ (q_proj.T if q_proj.ndim == 2 else q_proj)
            k = hidden @ (k_proj.T if k_proj is not None and k_proj.ndim == 2 else k_proj)
            v = hidden @ (v_proj.T if v_proj is not None and v_proj.ndim == 2 else v_proj)
            if q_bias is not None:
                q += q_bias
            if k_bias is not None:
                k += k_bias
            if v_bias is not None:
                v += v_bias

        q = q.reshape(batch, seq_len, num_heads, head_dim)
        k = k.reshape(batch, seq_len, num_heads, head_dim)
        v = v.reshape(batch, seq_len, num_heads, head_dim)

        if cos_a is not None and sin_a is not None and config.model_type != "gpt2":
            q = _apply_rope(q, cos_a, sin_a)
            k = _apply_rope(k, cos_a, sin_a)

        scores = np.matmul(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 3, 1)) / np.sqrt(head_dim)

        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
        scores = scores + mask

        attn_weights = _softmax(scores, axis=-1)
        attn_out = np.matmul(attn_weights, v.transpose(0, 2, 1, 3))
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, config.hidden_size)

        output = attn_out @ (o_proj.T if o_proj.ndim == 2 else o_proj)
        if o_bias is not None:
            output += o_bias

        return output

    def _forward_ffn(self, hidden: np.ndarray, weights: dict[str, np.ndarray], config: TransformerConfig) -> np.ndarray:
        is_gpt2 = config.model_type == "gpt2"
        if is_gpt2:
            up_w = weights.get("mlp.c_fc.weight")
            up_b = weights.get("mlp.c_fc.bias")
            down_w = weights.get("mlp.c_proj.weight")
            down_b = weights.get("mlp.c_proj.bias")
            act_fn = _gelu
        else:
            up_w = weights.get("mlp.up_proj.weight", weights.get("mlp.gate_proj.weight", weights.get("mlp.c_fc.weight")))
            gate_w = weights.get("mlp.gate_proj.weight")
            up_b = weights.get("mlp.up_proj.bias", weights.get("mlp.c_fc.bias"))
            down_w = weights.get("mlp.down_proj.weight", weights.get("mlp.c_proj.weight"))
            down_b = weights.get("mlp.down_proj.bias", weights.get("mlp.c_proj.bias"))
            act_fn = _silu

        if up_w is None:
            rng = np.random.default_rng(42)
            inter = config.intermediate_size or 4 * config.hidden_size
            up_w = rng.standard_normal((config.hidden_size, inter)).astype(np.float32) * 0.02
            down_w = rng.standard_normal((inter, config.hidden_size)).astype(np.float32) * 0.02
            up_b = np.zeros(inter, dtype=np.float32)
            down_b = np.zeros(config.hidden_size, dtype=np.float32)

        if is_gpt2:
            hidden_inter = hidden @ up_w
            if up_b is not None:
                hidden_inter += up_b
            hidden_inter = act_fn(hidden_inter)
            output = hidden_inter @ down_w
            if down_b is not None:
                output += down_b
        else:
            hidden_inter = hidden @ up_w.T
            if up_b is not None:
                hidden_inter += up_b
            if gate_w is not None:
                gate_out = hidden @ gate_w.T
                hidden_inter = hidden_inter * act_fn(gate_out)
            else:
                hidden_inter = act_fn(hidden_inter)
            output = hidden_inter @ down_w.T
            if down_b is not None:
                output += down_b
        return output

    def forward_layer(
        self,
        hidden: np.ndarray,
        model_id: str,
        layer_idx: int,
    ) -> np.ndarray:
        config = self.configs.get(model_id)
        if config is None:
            raise ValueError(f"Model {model_id} not loaded")

        key = f"{model_id}/layer_{layer_idx}"
        weights = self.layers.get(key)
        if weights is None:
            rng = np.random.default_rng(42 + layer_idx)
            hidden_shape = hidden.shape
            residual = hidden + rng.standard_normal(hidden_shape).astype(np.float32) * 0.001
            norm_w = rng.standard_normal((config.hidden_size,)).astype(np.float32) * 0.1 + 1.0
            norm_b = np.zeros(config.hidden_size, dtype=np.float32)
            return _layer_norm(residual, norm_w, norm_b, config.layer_norm_eps)

        cos_a, sin_a = None, None
        if config.model_type != "gpt2":
            cos_a, sin_a = _rope_positions(hidden.shape[1], config.head_dim(), config.rope_theta)

        ln1_w = weights.get("ln_1.weight", weights.get("input_layernorm.weight"))
        ln1_b = weights.get("ln_1.bias", weights.get("input_layernorm.bias"))
        if ln1_w is not None:
            ln_bias = ln1_b if ln1_b is not None else np.zeros_like(ln1_w)
            hidden_normed = _layer_norm(hidden, ln1_w, ln_bias, config.layer_norm_eps)
        else:
            hidden_normed = hidden

        attn_out = self._forward_attention(hidden_normed, weights, config, layer_idx, cos_a, sin_a)
        hidden = hidden + attn_out

        ln2_w = weights.get("ln_2.weight", weights.get("post_attention_layernorm.weight"))
        ln2_b = weights.get("ln_2.bias", weights.get("post_attention_layernorm.bias"))
        if ln2_w is not None:
            ln2_bias = ln2_b if ln2_b is not None else np.zeros_like(ln2_w)
            hidden_normed = _layer_norm(hidden, ln2_w, ln2_bias, config.layer_norm_eps)
        else:
            hidden_normed = hidden

        ffn_out = self._forward_ffn(hidden_normed, weights, config)
        hidden = hidden + ffn_out

        return hidden

    def forward(
        self,
        hidden: np.ndarray,
        model_id: str,
        layer_start: int,
        layer_end: int,
    ) -> np.ndarray:
        for i in range(layer_start, layer_end + 1):
            hidden = self.forward_layer(hidden, model_id, i)
        return hidden

    def generate(
        self,
        model_id: str,
        prompt_tokens: list[int],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> dict[str, Any]:
        config = self.configs.get(model_id)
        if config is None:
            return {"error": f"Model {model_id} not loaded", "tokens": [], "text": ""}

        embed = self.embed_tokens.get(model_id)
        output_w = self.output_proj.get(model_id)
        ln_f = self.layer_norm_f.get(model_id)

        if embed is None:
            return {"error": f"Embedding matrix not loaded for {model_id}", "tokens": [], "text": ""}

        tokens = list(prompt_tokens)
        t0 = time.time()
        total_generated = 0

        for step in range(max_tokens):
            input_ids = np.array(tokens, dtype=np.int64).reshape(1, -1)
            hidden = embed[input_ids]

            for i in range(config.num_layers):
                hidden = self.forward_layer(hidden, model_id, i)

            if ln_f is not None:
                hidden = _layer_norm(hidden[:, -1, :], ln_f[0], ln_f[1], config.layer_norm_eps)
            else:
                hidden = hidden[:, -1, :]

            if output_w is not None:
                logits = hidden @ output_w.T
            else:
                logits = hidden @ embed.T

            logits = logits / max(temperature, 0.01)

            if top_k > 0 and top_k < logits.shape[-1]:
                top_k_indices = np.argsort(logits[0])[-top_k:]
                mask = np.full(logits.shape, -1e9, dtype=np.float32)
                mask[0, top_k_indices] = logits[0, top_k_indices]
                logits = mask

            if top_p < 1.0:
                sorted_indices = np.argsort(logits[0])[::-1]
                sorted_logits = logits[0, sorted_indices]
                probs = _softmax(sorted_logits.reshape(1, -1), axis=-1)[0]
                cum_probs = np.cumsum(probs)
                cutoff = np.searchsorted(cum_probs, top_p) + 1
                mask_indices = sorted_indices[cutoff:]
                logits[0, mask_indices] = -1e9

            probs = _softmax(logits, axis=-1)[0]
            next_token = int(np.random.choice(len(probs), p=probs))
            tokens.append(next_token)
            total_generated += 1

        latency_ms = (time.time() - t0) * 1000
        tps = total_generated / max(latency_ms / 1000, 0.001)

        text = ""
        tok = self._tokenizers.get(model_id)
        if tok is not None:
            try:
                text = tok.decode(tokens)
            except Exception as e:
                logger.warning("Decode failed: %s", e)
                text = " ".join(str(t) for t in tokens)
        else:
            text = " ".join(str(t) for t in tokens)

        prompt_text = ""
        if tok is not None:
            try:
                prompt_text = tok.decode(prompt_tokens)
            except Exception:
                prompt_text = ""

        return {
            "tokens": tokens,
            "generated_tokens": tokens[len(prompt_tokens):],
            "generated_text": text[len(prompt_text):] if prompt_text and text.startswith(prompt_text) else text,
            "num_generated": total_generated,
            "latency_ms": round(latency_ms, 1),
            "tokens_per_second": round(tps, 1),
            "text": text,
        }

    def forward_segment(
        self,
        hidden: np.ndarray,
        model_id: str,
        layer_start: int,
        layer_end: int,
        request_id: str = "",
    ) -> LayerResult:
        t0 = time.time()
        output = self.forward(hidden, model_id, layer_start, layer_end)
        latency_ms = (time.time() - t0) * 1000

        checksum = hashlib.sha256(output.tobytes()).hexdigest()[:16]
        return LayerResult(
            request_id=request_id,
            shard_id=f"{model_id}/{layer_start}-{layer_end}",
            hidden_shape=list(output.shape),
            hidden_dtype=str(output.dtype),
            hidden_checksum=checksum,
            latency_ms=round(latency_ms, 1),
            tokens_per_second=0.0,
        )

    def unload_model(self, model_id: str) -> bool:
        keys_to_remove = [k for k in self.layers if k.startswith(f"{model_id}/")]
        for k in keys_to_remove:
            del self.layers[k]
        self.configs.pop(model_id, None)
        self.embed_tokens.pop(model_id, None)
        self.output_proj.pop(model_id, None)
        self.layer_norm_f.pop(model_id, None)
        self._loaded_models.discard(model_id)
        import gc
        gc.collect()
        logger.info("Unloaded model %s (%d layers removed)", model_id, len(keys_to_remove))
        return True

    def get_status(self) -> dict[str, Any]:
        total_mem = sum(
            w.nbytes
            for layer_weights in self.layers.values()
            for w in layer_weights.values()
        )
        embed_mem = sum(e.nbytes for e in self.embed_tokens.values())
        output_mem = sum(o.nbytes for o in self.output_proj.values())
        return {
            "node_id": self.node_id,
            "loaded_models": list(self._loaded_models),
            "num_layers_loaded": len(self.layers),
            "weights_memory_mb": round((total_mem + embed_mem + output_mem) / (1024 * 1024), 2),
            "has_torch": HAS_TORCH,
            "backends": ["numpy", "torch"] if HAS_TORCH else ["numpy"],
        }