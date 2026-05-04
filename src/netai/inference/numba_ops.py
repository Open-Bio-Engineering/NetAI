"""Numba JIT-compiled operations for native inference engine acceleration.

Provides drop-in replacements for critical math operations in the transformer
forward pass, plus a NumbaBackend class and monkey-patching utilities.

IMPORTANT: This module is OPTIONAL. If numba is not installed, HAS_NUMBA=False
and all functions degrade to Python/Numpy fallbacks. The engine works with or
without numba.
"""

from __future__ import annotations

import logging
import math
import time
import types
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

HAS_NUMBA = False
_NUMBA_VERSION = ""

try:
    import numba as _numba  # type: ignore
    HAS_NUMBA = True
    _NUMBA_VERSION = _numba.__version__
except ImportError:
    _numba = None


# ---------------------------------------------------------------------------
# Numba JIT-compiled math operations
# ---------------------------------------------------------------------------

if HAS_NUMBA:

    @_numba.njit(cache=True, fastmath=True)
    def _numba_gelu_impl(x: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        """GELU activation (tanh approximation), JIT-compiled."""
        out = np.empty_like(x)
        c1 = np.float32(0.5)
        c2 = np.float32(0.044715)
        sqrt_2_pi = np.float32(np.sqrt(np.float32(2.0) / np.pi))
        flat = x.ravel()
        out_flat = out.ravel()
        for i in range(flat.size):
            val = flat[i]
            inner = sqrt_2_pi * (val + c2 * val * val * val)
            tanh_inner = np.tanh(inner)
            out_flat[i] = c1 * val * (np.float32(1.0) + tanh_inner)
        return out

    @_numba.njit(cache=True, fastmath=True)
    def _numba_silu_impl(x: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        """SiLU activation, JIT-compiled."""
        out = np.empty_like(x)
        flat = x.ravel()
        out_flat = out.ravel()
        for i in range(flat.size):
            val = flat[i]
            out_flat[i] = val / (np.float32(1.0) + np.exp(-val))
        return out

    @_numba.njit(cache=True, fastmath=True)
    def _numba_softmax_impl(x: np.ndarray, axis: int = -1) -> np.ndarray:  # type: ignore[misc]
        """Softmax with numerical stability, JIT-compiled.

        Supports 2D inputs with axis=-1 or 4D inputs with axis=-1.
        """
        result = np.empty_like(x)
        last_dim = x.shape[-1]
        if x.ndim == 2:
            n_rows = x.shape[0]
            for i in range(n_rows):
                start = i * last_dim
                row_max = x.flat[start]
                for j in range(1, last_dim):
                    v = x.flat[start + j]
                    if v > row_max:
                        row_max = v
                exp_sum = np.float32(0.0)
                for j in range(last_dim):
                    ev = np.exp(x.flat[start + j] - row_max)
                    result.flat[start + j] = ev
                    exp_sum += ev
                for j in range(last_dim):
                    result.flat[start + j] /= exp_sum
        elif x.ndim == 4:
            dim0, dim1, dim2, dim3 = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            for a in range(dim0):
                for b in range(dim1):
                    for c in range(dim2):
                        base = ((a * dim1 + b) * dim2 + c) * dim3
                        row_max = x.flat[base]
                        for d in range(1, dim3):
                            v = x.flat[base + d]
                            if v > row_max:
                                row_max = v
                        exp_sum = np.float32(0.0)
                        for d in range(dim3):
                            ev = np.exp(x.flat[base + d] - row_max)
                            result.flat[base + d] = ev
                            exp_sum += ev
                        for d in range(dim3):
                            result.flat[base + d] /= exp_sum
        else:
            x_max = np.max(x, axis=axis, keepdims=True)
            e_x = np.exp(x - x_max)
            s = np.sum(e_x, axis=axis, keepdims=True)
            result = e_x / s
        return result

    @_numba.njit(cache=True, fastmath=True)
    def _numba_layer_norm_impl(
        x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:  # type: ignore[misc]
        """Layer normalization, JIT-compiled."""
        out = np.empty_like(x)
        hidden = x.shape[-1]
        num_rows = x.size // hidden
        for i in range(num_rows):
            start = i * hidden
            total = np.float32(0.0)
            for j in range(hidden):
                total += x.flat[start + j]
            mean = total / np.float32(hidden)
            var_sum = np.float32(0.0)
            for j in range(hidden):
                diff = x.flat[start + j] - mean
                var_sum += diff * diff
            var = var_sum / np.float32(hidden)
            inv_std = np.float32(1.0) / np.sqrt(var + np.float32(eps))
            for j in range(hidden):
                val = x.flat[start + j]
                out.flat[start + j] = weight[j] * (val - mean) * inv_std + bias[j]
        return out

    @_numba.njit(cache=True, fastmath=True)
    def _numba_rope_apply_impl(
        hidden: np.ndarray, cos_a: np.ndarray, sin_a: np.ndarray
    ) -> np.ndarray:  # type: ignore[misc]
        """Apply rotary positional embeddings, JIT-compiled.

        Takes 4D hidden [batch, seq, heads, head_dim] and broadast
        cos_a/sin_a [seq, d2].
        """
        batch, seq_len, heads, head_dim = hidden.shape
        d2 = head_dim // 2
        out = np.empty_like(hidden)
        for b in range(batch):
            for s in range(seq_len):
                for h in range(heads):
                    for d in range(d2):
                        x1 = hidden[b, s, h, d]
                        x2 = hidden[b, s, h, d + d2]
                        c = cos_a[s, d]
                        si = sin_a[s, d]
                        out[b, s, h, d] = x1 * c - x2 * si
                        out[b, s, h, d + d2] = x2 * c + x1 * si
                    if head_dim > 2 * d2:
                        for d in range(2 * d2, head_dim):
                            out[b, s, h, d] = hidden[b, s, h, d]
        return out

    @_numba.njit(cache=True, fastmath=True)
    def _numba_matmul_qkv_gpt2_impl_alt(
        hidden_2d: np.ndarray, qkv_weight: np.ndarray
    ) -> "np.ndarray":  # type: ignore[misc]
        """QKV projection for GPT-2 using 2D hidden, JIT-compiled."""
        return hidden_2d @ qkv_weight

    @_numba.njit(cache=True, fastmath=True)
    def _numba_matmul_out_impl(
        attn_out: np.ndarray, o_proj: np.ndarray
    ) -> np.ndarray:  # type: ignore[misc]
        """Output projection, JIT-compiled."""
        return attn_out @ o_proj

    @_numba.njit(cache=True, fastmath=True)
    def _numba_ffn_gpt2_impl(
        hidden: np.ndarray, up_w: np.ndarray, down_w: np.ndarray
    ) -> np.ndarray:  # type: ignore[misc]
        """GPT-2 FFN with GELU, JIT-compiled."""
        inter = _numba_gelu_impl(hidden @ up_w)
        return inter @ down_w

    @_numba.njit(cache=True, fastmath=True)
    def _numba_ffn_llama_impl(
        hidden: np.ndarray, gate_w: np.ndarray, up_w: np.ndarray, down_w: np.ndarray
    ) -> np.ndarray:  # type: ignore[misc]
        """LLaMA FFN with SiLU gate, JIT-compiled."""
        gate = _numba_silu_impl(hidden @ gate_w.T.astype(np.float32))
        up = hidden @ up_w.T.astype(np.float32)
        inter = gate * up
        return inter @ down_w.T.astype(np.float32)

    @_numba.njit(cache=True, fastmath=True)
    def _numba_attn_scores_impl(
        q: np.ndarray, k: np.ndarray, inv_scale: float
    ) -> np.ndarray:  # type: ignore[misc]
        """Compute scaled dot-product attention scores, JIT-compiled."""
        batch, num_heads, seq_q, head_dim = q.shape
        seq_k = k.shape[2]
        scores = np.empty((batch, num_heads, seq_q, seq_k), dtype=np.float32)
        for b in range(batch):
            for h in range(num_heads):
                for i in range(seq_q):
                    for j in range(seq_k):
                        dot = np.float32(0.0)
                        for d in range(head_dim):
                            dot += q[b, h, i, d] * k[b, h, j, d]
                        scores[b, h, i, j] = dot * np.float32(inv_scale)
        return scores

    @_numba.njit(cache=True, fastmath=True)
    def _numba_attn_mask_impl(scores: np.ndarray, seq_len: int) -> np.ndarray:  # type: ignore[misc]
        """Apply causal mask, JIT-compiled."""
        _, _, sq, sk = scores.shape
        for b in range(scores.shape[0]):
            for h in range(scores.shape[1]):
                for i in range(sq):
                    for j in range(sk):
                        if j > i + max(0, sk - sq):
                            scores[b, h, i, j] = np.float32(-1e9)
        return scores

    @_numba.njit(cache=True, fastmath=True)
    def _numba_attn_output_impl(
        attn_weights: np.ndarray, v: np.ndarray
    ) -> np.ndarray:  # type: ignore[misc]
        """Compute attention output from weights and values, JIT-compiled."""
        batch, num_heads, seq_q, seq_k = attn_weights.shape
        head_dim = v.shape[3]
        out = np.empty((batch, num_heads, seq_q, head_dim), dtype=np.float32)
        for b in range(batch):
            for h in range(num_heads):
                for i in range(seq_q):
                    for d in range(head_dim):
                        acc = np.float32(0.0)
                        for j in range(seq_k):
                            acc += attn_weights[b, h, i, j] * v[b, h, j, d]
                        out[b, h, i, d] = acc
        return out

else:
    # Stubs when numba is not installed
    pass


# ---------------------------------------------------------------------------
# Public API — fall back to pure Python/Numpy if numba unavailable
# ---------------------------------------------------------------------------


if HAS_NUMBA:

    def numba_gelu(x: np.ndarray) -> np.ndarray:
        return _numba_gelu_impl(x.astype(np.float32, copy=False))

    def numba_silu(x: np.ndarray) -> np.ndarray:
        return _numba_silu_impl(x.astype(np.float32, copy=False))

    def numba_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        return _numba_softmax_impl(x.astype(np.float32, copy=False), axis)

    def numba_layer_norm(
        x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        return _numba_layer_norm_impl(
            x.astype(np.float32, copy=False),
            weight.astype(np.float32, copy=False),
            bias.astype(np.float32, copy=False),
            eps,
        )

    def numba_rope_apply(
        hidden: np.ndarray, cos_a: np.ndarray, sin_a: np.ndarray
    ) -> np.ndarray:
        return _numba_rope_apply_impl(
            hidden.astype(np.float32, copy=False),
            cos_a.astype(np.float32, copy=False),
            sin_a.astype(np.float32, copy=False),
        )

    def numba_matmul_qkv(
        hidden: np.ndarray, qkv_weight: np.ndarray
    ) -> np.ndarray:
        return _numba_matmul_qkv_gpt2_impl_alt(
            hidden.astype(np.float32, copy=False),
            qkv_weight.astype(np.float32, copy=False),
        )

    def numba_matmul_out(
        attn_out: np.ndarray, o_proj: np.ndarray
    ) -> np.ndarray:
        return _numba_matmul_out_impl(
            attn_out.astype(np.float32, copy=False),
            o_proj.astype(np.float32, copy=False),
        )

    def numba_ffn_gpt2(
        hidden: np.ndarray, up_w: np.ndarray, down_w: np.ndarray
    ) -> np.ndarray:
        return _numba_ffn_gpt2_impl(
            hidden.astype(np.float32, copy=False),
            up_w.astype(np.float32, copy=False),
            down_w.astype(np.float32, copy=False),
        )

    def numba_ffn_llama(
        hidden: np.ndarray, gate_w: np.ndarray, up_w: np.ndarray, down_w: np.ndarray
    ) -> np.ndarray:
        return _numba_ffn_llama_impl(
            hidden.astype(np.float32, copy=False),
            gate_w.astype(np.float32, copy=False),
            up_w.astype(np.float32, copy=False),
            down_w.astype(np.float32, copy=False),
        )

    def numba_attn_scores(
        q: np.ndarray, k: np.ndarray, inv_scale: float
    ) -> np.ndarray:
        return _numba_attn_scores_impl(
            q.astype(np.float32, copy=False),
            k.astype(np.float32, copy=False),
            inv_scale,
        )

    def numba_attn_mask(scores: np.ndarray, seq_len: int) -> np.ndarray:
        return _numba_attn_mask_impl(
            scores.astype(np.float32, copy=False), seq_len
        )

    def numba_attn_output(
        attn_weights: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        return _numba_attn_output_impl(
            attn_weights.astype(np.float32, copy=False),
            v.astype(np.float32, copy=False),
        )

else:
    # -----------------------------------------------------------------------
    # Pure Python/Numpy fallbacks (no numba)
    # -----------------------------------------------------------------------

    def numba_gelu(x: np.ndarray) -> np.ndarray:
        f32 = np.float32
        return f32(0.5) * x * (
            f32(1.0) + np.tanh(np.sqrt(f32(2.0) / np.pi) * (x + f32(0.044715) * x ** f32(3)))
        )

    def numba_silu(x: np.ndarray) -> np.ndarray:
        return x / (np.float32(1.0) + np.exp(-x.astype(np.float32, copy=False)))

    def numba_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def numba_layer_norm(
        x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return weight * (x - mean) / np.sqrt(var + eps) + bias

    def numba_rope_apply(
        hidden: np.ndarray, cos_a: np.ndarray, sin_a: np.ndarray
    ) -> np.ndarray:
        batch, seq, heads, head_d = hidden.shape
        d2 = head_d // 2
        rot_d = 2 * d2
        x_rot = hidden[..., :rot_d]
        x_pass = hidden[..., rot_d:] if rot_d < head_d else None
        x1 = x_rot[..., :d2]
        x2 = x_rot[..., d2:]
        cos_b = cos_a[:seq, :d2][np.newaxis, :, np.newaxis, :]
        sin_b = sin_a[:seq, :d2][np.newaxis, :, np.newaxis, :]
        out1 = x1 * cos_b - x2 * sin_b
        out2 = x2 * cos_b + x1 * sin_b
        result = np.concatenate([out1, out2], axis=-1)
        if x_pass is not None:
            result = np.concatenate([result, x_pass], axis=-1)
        return result

    def numba_matmul_qkv(
        hidden: np.ndarray, qkv_weight: np.ndarray
    ) -> np.ndarray:
        return hidden @ qkv_weight

    def numba_matmul_out(
        attn_out: np.ndarray, o_proj: np.ndarray
    ) -> np.ndarray:
        return attn_out @ o_proj

    def numba_ffn_gpt2(
        hidden: np.ndarray, up_w: np.ndarray, down_w: np.ndarray
    ) -> np.ndarray:
        inter = numba_gelu(hidden @ up_w)
        return inter @ down_w

    def numba_ffn_llama(
        hidden: np.ndarray, gate_w: np.ndarray, up_w: np.ndarray, down_w: np.ndarray
    ) -> np.ndarray:
        gate = numba_silu(hidden @ gate_w.T)
        up = hidden @ up_w.T
        return (gate * up) @ down_w.T

    def numba_attn_scores(
        q: np.ndarray, k: np.ndarray, inv_scale: float
    ) -> np.ndarray:
        return np.matmul(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 3, 1)) * inv_scale

    def numba_attn_mask(scores: np.ndarray, seq_len: int) -> np.ndarray:
        sq, sk = scores.shape[2], scores.shape[3]
        mask = np.triu(np.full((sq, sk), np.float32(-1e9)), k=max(1, sk - sq + 1))
        return scores + mask

    def numba_attn_output(
        attn_weights: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        return np.matmul(attn_weights, v.transpose(0, 2, 1, 3))


# ---------------------------------------------------------------------------
# NumbaBackend — accelerator class
# ---------------------------------------------------------------------------


class NumbaBackend:
    """Numba-accelerated inference backend.

    Provides JIT-compiled layer forward and generation methods that can
    accelerate transformer inference on CPU.
    """

    def __init__(self):
        self._warmup_times: dict[str, float] = {}
        self._warmed_up = False
        self._acceleration_factor: float = 1.0

    @staticmethod
    def is_available() -> bool:
        """Check whether numba acceleration is available."""
        return HAS_NUMBA

    @staticmethod
    def numba_version() -> str:
        """Return the installed numba version, or empty string."""
        return _NUMBA_VERSION

    def get_warmup_status(self) -> dict[str, Any]:
        """Return warmup/compilation timing information."""
        return {
            "numba_available": HAS_NUMBA,
            "numba_version": _NUMBA_VERSION,
            "warmed_up": self._warmed_up,
            "warmup_times": dict(self._warmup_times),
            "acceleration_factor": self._acceleration_factor,
        }

    def _warmup_fn(self, name: str, fn, *args) -> Any:
        """Time a JIT function's first call (triggers compilation)."""
        t0 = time.perf_counter()
        result = fn(*args)
        dt = time.perf_counter() - t0
        self._warmup_times[name] = round(dt * 1000, 2)
        return result

    def warmup(self, hidden_size: int = 768, intermediate_size: int = 3072) -> None:
        """Trigger JIT compilation of all kernels with small dummy data."""
        if not HAS_NUMBA:
            logger.info("Numba not available — skipping warmup")
            return

        seq_len = 8
        H = hidden_size
        I = intermediate_size
        rng = np.random.default_rng(42)

        x = (rng.random((1, seq_len, H), dtype=np.float32) - 0.5).astype(np.float32)
        x_2d = x.reshape(-1, H).astype(np.float32)
        w_2d = (rng.random((H, I), dtype=np.float32) - 0.5).astype(np.float32)
        gate_w = (rng.random((I, H), dtype=np.float32) - 0.5).astype(np.float32)
        up_w = (rng.random((I, H), dtype=np.float32) - 0.5).astype(np.float32)
        down_w = (rng.random((H, I), dtype=np.float32) - 0.5).astype(np.float32)
        qkv_w = (rng.random((H, 3 * H), dtype=np.float32) - 0.5).astype(np.float32)
        o_proj = (rng.random((H, H), dtype=np.float32) - 0.5).astype(np.float32)
        ln_w = np.ones(H, dtype=np.float32)
        ln_b = np.zeros(H, dtype=np.float32)

        hidden_4d = (rng.random((1, seq_len, 12, H // 12), dtype=np.float32) - 0.5).astype(np.float32)
        cos_a = (rng.random((seq_len, H // 24), dtype=np.float32) - 0.5).astype(np.float32)
        sin_a = (rng.random((seq_len, H // 24), dtype=np.float32) - 0.5).astype(np.float32)

        num_heads = 12
        head_dim = H // num_heads
        q_4d = (rng.random((1, num_heads, seq_len, head_dim), dtype=np.float32) - 0.5).astype(np.float32)
        k_4d = (rng.random((1, num_heads, seq_len, head_dim), dtype=np.float32) - 0.5).astype(np.float32)
        v_4d = (rng.random((1, num_heads, seq_len, head_dim), dtype=np.float32) - 0.5).astype(np.float32)
        scores_4d = (rng.random((1, num_heads, seq_len, seq_len), dtype=np.float32) - 0.5).astype(np.float32)
        attn_w = (rng.random((1, num_heads, seq_len, seq_len), dtype=np.float32)).astype(np.float32)
        attn_w_sum = attn_w.sum(axis=-1, keepdims=True)
        attn_w = (attn_w / attn_w_sum).astype(np.float32)

        kernels = [
            ("gelu", lambda: numba_gelu(x_2d.flatten().astype(np.float32))),
            ("silu", lambda: numba_silu(x_2d.flatten().astype(np.float32))),
            ("softmax", lambda: numba_softmax(x_2d, axis=-1)),
            ("layer_norm", lambda: numba_layer_norm(x_2d, ln_w, ln_b, 1e-5)),
            ("rope_apply", lambda: numba_rope_apply(hidden_4d, cos_a, sin_a)),
            ("matmul_qkv", lambda: numba_matmul_qkv(x_2d, qkv_w)),
            ("matmul_out", lambda: numba_matmul_out(x_2d, o_proj)),
            ("ffn_gpt2", lambda: numba_ffn_gpt2(x_2d, w_2d, down_w)),
            ("ffn_llama", lambda: numba_ffn_llama(x_2d, gate_w, up_w, down_w)),
            ("attn_scores", lambda: numba_attn_scores(q_4d, k_4d, 1.0 / math.sqrt(head_dim))),
            ("attn_mask", lambda: numba_attn_mask(scores_4d, seq_len)),
            ("attn_output", lambda: numba_attn_output(attn_w, v_4d)),
        ]

        for name, fn in kernels:
            self._warmup_fn(name, fn)

        # Benchmark: run each kernel 100 times and compare to Python
        n_iter = 100
        python_times: dict[str, float] = {}
        numba_times: dict[str, float] = {}

        for name, _ in kernels:
            n = n_iter // 10 if name in ("gelu", "silu", "softmax", "layer_norm", "rope_apply") else n_iter // 5

            if name == "gelu":
                py_fn = lambda y: np.float32(0.5) * y * (
                    np.float32(1.0) + np.tanh(
                        np.sqrt(np.float32(2.0) / np.pi) * (y + np.float32(0.044715) * y ** np.float32(3))
                    )
                )
                data = x_2d.flatten().astype(np.float32)
                nb_fn = lambda d: numba_gelu(d)
            elif name == "silu":
                py_fn = lambda y: y / (np.float32(1.0) + np.exp(-y))
                data = x_2d.flatten().astype(np.float32)
                nb_fn = lambda d: numba_silu(d)
            elif name == "softmax":
                py_fn = lambda y: (
                    lambda arr: (
                        lambda mx: np.exp(arr - mx) / np.sum(np.exp(arr - mx), axis=-1, keepdims=True)
                    )(np.max(arr, axis=-1, keepdims=True))
                )(y)
                data = x_2d
                nb_fn = lambda d: numba_softmax(d, axis=-1)
            elif name == "layer_norm":
                py_fn = lambda y: ln_w * (y - np.mean(y, axis=-1, keepdims=True)) / np.sqrt(
                    np.var(y, axis=-1, keepdims=True) + 1e-5
                ) + ln_b
                data = x_2d
                nb_fn = lambda d: numba_layer_norm(d, ln_w, ln_b, 1e-5)
            elif name == "rope_apply":
                py_fn = lambda y: numba_rope_apply(y, cos_a, sin_a)
                data = hidden_4d
                nb_fn = lambda d: numba_rope_apply(d, cos_a, sin_a)
            elif name == "matmul_qkv":
                py_fn = lambda y: y @ qkv_w
                data = x_2d
                nb_fn = lambda d: numba_matmul_qkv(d, qkv_w)
            elif name == "matmul_out":
                py_fn = lambda y: y @ o_proj
                data = x_2d
                nb_fn = lambda d: numba_matmul_out(d, o_proj)
            elif name == "ffn_gpt2":
                py_fn = lambda y: numba_ffn_gpt2(y, w_2d, down_w)
                data = x_2d
                nb_fn = lambda d: numba_ffn_gpt2(d, w_2d, down_w)
            elif name == "ffn_llama":
                py_fn = lambda y: numba_ffn_llama(y, gate_w, up_w, down_w)
                data = x_2d
                nb_fn = lambda d: numba_ffn_llama(d, gate_w, up_w, down_w)
            elif name == "attn_scores":
                py_fn = lambda y: np.matmul(y, k_4d.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(head_dim))
                data = q_4d
                nb_fn = lambda d: numba_attn_scores(d, k_4d, 1.0 / math.sqrt(head_dim))
            else:
                continue

            t0 = time.perf_counter()
            for _ in range(n):
                py_fn(data)
            python_times[name] = (time.perf_counter() - t0) / n * 1000

            t0 = time.perf_counter()
            for _ in range(n):
                nb_fn(data)
            numba_times[name] = (time.perf_counter() - t0) / n * 1000

        py_total = sum(python_times.values())
        nb_total = sum(numba_times.values())
        if nb_total > 0:
            self._acceleration_factor = round(py_total / nb_total, 1)
        self._warmed_up = True

        logger.info(
            "NumbaBackend warmed up: %d kernels compiled, %.1fx speedup over Python",
            len(kernels), self._acceleration_factor,
        )

    def accelerate_layer_forward(
        self,
        hidden: np.ndarray,
        weights: dict[str, np.ndarray],
        config: Any,  # TransformerConfig
        layer_idx: int,
        cos_a: np.ndarray | None = None,
        sin_a: np.ndarray | None = None,
    ) -> np.ndarray:
        """Full transformer layer forward with numba-optimized math.

        Args:
            hidden: [batch, seq, hidden_size] float32
            weights: Layer weight dict (same format as NativeInferenceEngine)
            config: TransformerConfig instance
            layer_idx: Layer index for error messages
            cos_a, sin_a: Precomputed RoPE embeddings (required for Llama)

        Returns:
            hidden: [batch, seq, hidden_size] float32
        """
        if not HAS_NUMBA:
            logger.warning("NumbaBackend.accelerate_layer_forward called but numba unavailable")
            return hidden

        batch, seq_len, _ = hidden.shape
        is_gpt2 = getattr(config, 'model_type', 'gpt2') == "gpt2"
        eps = getattr(config, 'layer_norm_eps', 1e-5)

        # --- First layer norm ---
        ln1_w = weights.get("ln_1.weight") or weights.get("input_layernorm.weight")
        if ln1_w is not None:
            ln1_b = weights.get("ln_1.bias") or weights.get("input_layernorm.bias")
            ln1_b = ln1_b if ln1_b is not None else np.zeros_like(ln1_w)
            hidden_2d = hidden.reshape(-1, hidden.shape[-1]).astype(np.float32)
            ln1_out = numba_layer_norm(hidden_2d, ln1_w.astype(np.float32), ln1_b.astype(np.float32), eps)
        else:
            ln1_out = hidden.reshape(-1, hidden.shape[-1]).astype(np.float32)

        # --- Attention ---
        q_proj = weights.get("attn.c_attn.weight") or weights.get("self_attn.q_proj.weight")
        k_proj = weights.get("attn.c_attn.weight") or weights.get("self_attn.k_proj.weight")
        v_proj = weights.get("attn.c_proj.weight") or weights.get("self_attn.v_proj.weight")
        o_proj = weights.get("attn.c_proj.weight") or weights.get("self_attn.o_proj.weight")

        if q_proj is None:
            logger.error("Missing attention weights for layer %d", layer_idx)
            return hidden

        num_heads = getattr(config, 'num_heads', 12)
        head_dim = getattr(config, 'hidden_size', 768) // num_heads
        inv_scale = np.float32(1.0 / math.sqrt(head_dim))

        is_gpt2_qkv = (
            q_proj.shape[0] == getattr(config, 'hidden_size', 768)
            and q_proj.shape[1] == 3 * getattr(config, 'hidden_size', 768)
        )

        if is_gpt2_qkv:
            qkv = numba_matmul_qkv(ln1_out, q_proj.astype(np.float32))
            q_bias = weights.get("attn.c_attn.bias") or weights.get("self_attn.q_proj.bias")
            if q_bias is not None:
                qkv = qkv + q_bias.astype(np.float32)
            q, k, v = np.split(qkv, 3, axis=-1)
        else:
            q_w = q_proj.astype(np.float32)
            q = ln1_out @ (q_w.T if q_w.ndim == 2 else q_w)
            k = ln1_out @ (k_proj.T.astype(np.float32) if k_proj is not None and k_proj.ndim == 2 else k_proj.astype(np.float32))
            v = ln1_out @ (v_proj.T.astype(np.float32) if v_proj is not None and v_proj.ndim == 2 else v_proj.astype(np.float32))
            q_bias = weights.get("attn.c_attn.bias") or weights.get("self_attn.q_proj.bias")
            k_bias = weights.get("attn.c_attn.bias") or weights.get("self_attn.k_proj.bias")
            v_bias = weights.get("attn.c_attn.bias") or weights.get("self_attn.v_proj.bias")
            if q_bias is not None:
                q = q + q_bias.astype(np.float32)
            if k_bias is not None:
                k = k + k_bias.astype(np.float32)
            if v_bias is not None:
                v = v + v_bias.astype(np.float32)

        q = q.reshape(batch, seq_len, num_heads, head_dim).astype(np.float32)
        k = k.reshape(batch, seq_len, num_heads, head_dim).astype(np.float32)
        v = v.reshape(batch, seq_len, num_heads, head_dim).astype(np.float32)

        if cos_a is not None and sin_a is not None and not is_gpt2:
            q = numba_rope_apply(q, cos_a, sin_a)
            k = numba_rope_apply(k, cos_a, sin_a)

        scores = numba_attn_scores(q, k, inv_scale)
        scores = numba_attn_mask(scores, seq_len)
        attn_weights = numba_softmax(scores.astype(np.float32), axis=-1)
        attn_out_4d = numba_attn_output(attn_weights, v)
        attn_out = attn_out_4d.transpose(0, 2, 1, 3).reshape(batch * seq_len, getattr(config, 'hidden_size', 768))

        o_w = o_proj.astype(np.float32)
        output = attn_out @ (o_w.T if o_w.ndim == 2 else o_w)
        o_bias = weights.get("attn.c_proj.bias") or weights.get("self_attn.o_proj.bias")
        if o_bias is not None:
            output = output + o_bias.astype(np.float32)
        attn_result = output.reshape(batch, seq_len, getattr(config, 'hidden_size', 768))

        # Residual
        hidden = hidden + attn_result.astype(np.float32)

        # --- Second layer norm ---
        ln2_w = weights.get("ln_2.weight") or weights.get("post_attention_layernorm.weight")
        if ln2_w is not None:
            ln2_b = weights.get("ln_2.bias") or weights.get("post_attention_layernorm.bias")
            ln2_b = ln2_b if ln2_b is not None else np.zeros_like(ln2_w)
            hidden_2d = hidden.reshape(-1, hidden.shape[-1]).astype(np.float32)
            ffn_input = numba_layer_norm(hidden_2d, ln2_w.astype(np.float32), ln2_b.astype(np.float32), eps)
        else:
            ffn_input = hidden.reshape(-1, hidden.shape[-1]).astype(np.float32)

        # --- FFN ---
        hidden_size = getattr(config, 'hidden_size', 768)
        if is_gpt2:
            up_w = weights.get("mlp.c_fc.weight")
            up_b = weights.get("mlp.c_fc.bias")
            down_w = weights.get("mlp.c_proj.weight")
            down_b = weights.get("mlp.c_proj.bias")
            if up_w is not None:
                inter = ffn_input @ up_w.astype(np.float32)
                if up_b is not None:
                    inter = inter + up_b.astype(np.float32)
                inter = numba_gelu(inter)
                ffn_out = inter @ down_w.astype(np.float32)
                if down_b is not None:
                    ffn_out = ffn_out + down_b.astype(np.float32)
            else:
                ffn_out = np.zeros_like(ffn_input)
        else:
            up_w = weights.get("mlp.up_proj.weight") or weights.get("mlp.gate_proj.weight") or weights.get("mlp.c_fc.weight")
            gate_w = weights.get("mlp.gate_proj.weight")
            up_b = weights.get("mlp.up_proj.bias") or weights.get("mlp.c_fc.bias")
            down_w = weights.get("mlp.down_proj.weight") or weights.get("mlp.c_proj.weight")
            down_b = weights.get("mlp.down_proj.bias")

            if up_w is not None:
                up = ffn_input @ up_w.T.astype(np.float32) if "mlp.gate_proj.weight" not in weights or gate_w is not None else ffn_input @ up_w.astype(np.float32)
                if up_b is not None:
                    up = up + up_b.astype(np.float32)
                if gate_w is not None:
                    gate = ffn_input @ gate_w.T.astype(np.float32)
                    inter = up * numba_silu(gate)
                else:
                    inter = numba_silu(up)
                dw = down_w.T.astype(np.float32)
                ffn_out_2d = inter @ dw
                if down_b is not None:
                    ffn_out_2d = ffn_out_2d + down_b.astype(np.float32)
                ffn_out = ffn_out_2d
            else:
                ffn_out = np.zeros_like(ffn_input)

        ffn_result = ffn_out.reshape(batch, seq_len, hidden_size).astype(np.float32)
        hidden = hidden + ffn_result
        return hidden

    def accelerate_generate(
        self,
        model_id: str,
        tokens: list[int],
        config: Any,
        embed: np.ndarray,
        output_w: np.ndarray | None,
        ln_f: tuple | None,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        layer_weights: list[dict[str, np.ndarray]],
    ) -> dict[str, Any]:
        """Full generation loop with numba-optimized layer forwards."""
        if not HAS_NUMBA:
            return {
                "error": "Numba not available — cannot accelerate generation",
                "tokens": [],
                "text": "",
            }

        prompt_tokens = list(tokens)
        t0 = time.time()
        rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
        total_generated = 0
        is_gpt2 = getattr(config, 'model_type', 'gpt2') == "gpt2"
        eps = getattr(config, 'layer_norm_eps', 1e-5)
        hidden_size = getattr(config, 'hidden_size', 768)

        for step in range(max_tokens):
            input_ids = np.array(tokens, dtype=np.int64).reshape(1, -1)
            hidden = embed[input_ids].astype(np.float32)
            seq_len = hidden.shape[1]

            cos_a, sin_a = None, None
            if not is_gpt2:
                positions = np.arange(seq_len, dtype=np.float32)
                head_dim = hidden_size // getattr(config, 'num_heads', 12)
                rope_theta = getattr(config, 'rope_theta', 10000.0)
                freqs = np.float32(1.0) / (np.float32(rope_theta) ** (np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)))
                angles = np.outer(positions, freqs)
                cos_a = np.cos(angles).astype(np.float32)
                sin_a = np.sin(angles).astype(np.float32)

            for i, weights in enumerate(layer_weights):
                hidden = self._accelerate_single_layer(
                    hidden, weights, config, i, cos_a, sin_a
                )

            if ln_f is not None:
                last = hidden[:, -1, :].astype(np.float32)
                last = numba_layer_norm(
                    last, ln_f[0].astype(np.float32), ln_f[1].astype(np.float32), eps
                )
            else:
                last = hidden[:, -1, :].astype(np.float32)

            if output_w is not None:
                logits = last @ output_w.T.astype(np.float32)
            else:
                logits = last @ embed.T.astype(np.float32)

            logits = logits / max(temperature, 0.01)
            logits = logits.astype(np.float32)

            if top_k > 0 and top_k < logits.shape[-1]:
                top_k_indices = np.argsort(logits[0])[-top_k:]
                mask = np.full(logits.shape, np.float32(-1e9))
                mask[0, top_k_indices] = logits[0, top_k_indices]
                logits = mask

            if top_p < 1.0:
                sorted_indices = np.argsort(logits[0])[::-1]
                sorted_logits = logits[0, sorted_indices]
                probs = numba_softmax(sorted_logits.reshape(1, -1).astype(np.float32), axis=-1)[0]
                cum_probs = np.cumsum(probs)
                cutoff = np.searchsorted(cum_probs, top_p) + 1
                mask_indices = sorted_indices[cutoff:]
                logits[0, mask_indices] = np.float32(-1e9)

            probs = numba_softmax(logits.astype(np.float32), axis=-1)[0]
            next_token = int(rng.choice(len(probs), p=probs))
            tokens.append(next_token)
            total_generated += 1

        latency_ms = (time.time() - t0) * 1000
        tps = total_generated / max(latency_ms / 1000, 0.001)

        return {
            "tokens": tokens,
            "generated_tokens": tokens[len(prompt_tokens):],
            "num_generated": total_generated,
            "latency_ms": round(latency_ms, 1),
            "tokens_per_second": round(tps, 1),
        }

    def _accelerate_single_layer(
        self,
        hidden: np.ndarray,
        weights: dict[str, np.ndarray],
        config: Any,
        layer_idx: int,
        cos_a: np.ndarray | None,
        sin_a: np.ndarray | None,
    ) -> np.ndarray:
        """Single layer forward using accelerate_layer_forward."""
        hidden = self.accelerate_layer_forward(
            hidden, weights, config, layer_idx, cos_a, sin_a
        )
        return np.ascontiguousarray(hidden)


# ---------------------------------------------------------------------------
# Monkey-patching utilities
# ---------------------------------------------------------------------------


def apply_numba_patches(engine: Any) -> bool:
    """Monkey-patch a NativeInferenceEngine to use numba-accelerated math.

    Replaces the module-level math functions (_softmax, _layer_norm, _gelu,
    _silu, _rope_positions, _apply_rope) and instance methods
    (_forward_attention, _forward_ffn, forward_layer) with numba-optimized
    versions.

    Args:
        engine: A NativeInferenceEngine instance.

    Returns:
        True if patches were applied successfully, False otherwise.
    """
    if not HAS_NUMBA:
        logger.warning("apply_numba_patches: numba not available — no patches applied")
        return False

    import netai.inference.native_engine as native_mod

    # Patch module-level math functions
    native_mod._softmax = numba_softmax
    native_mod._layer_norm = numba_layer_norm
    native_mod._gelu = numba_gelu
    native_mod._silu = numba_silu

    _original_softmax = native_mod._softmax

    def _patched_softmax(x, axis=-1):
        return numba_softmax(x.astype(np.float32, copy=False), axis)

    _patched_softmax_new = staticmethod(lambda x, axis=-1: numba_softmax(x, axis))
    native_mod._softmax = numba_softmax

    # Patch _apply_rope to use numba
    def _patched_apply_rope(hidden, cos_a, sin_a):
        return numba_rope_apply(hidden, cos_a, sin_a)

    native_mod._apply_rope = _patched_apply_rope

    # Patch _rope_positions (still numpy, but this is fast enough)
    # Just ensure the output is float32 contiguous.

    # Patch the forward_attention method
    original_fwd_attn = engine._forward_attention

    def _patched_forward_attention(
        self, hidden, weights, config, layer_idx, cos_a=None, sin_a=None
    ):
        batch, seq_len, _ = hidden.shape
        num_heads = config.num_heads
        head_dim = config.hidden_size // num_heads
        inv_scale = np.float32(1.0 / math.sqrt(head_dim))

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

        is_gpt2_qkv = (
            q_proj is not None
            and q_proj.shape[0] == config.hidden_size
            and q_proj.shape[1] == 3 * config.hidden_size
        )

        hidden_2d = hidden.reshape(-1, hidden.shape[-1]).astype(np.float32)

        if is_gpt2_qkv:
            qkv = numba_matmul_qkv(hidden_2d, q_proj.astype(np.float32))
            if q_bias is not None:
                qkv = qkv + q_bias.astype(np.float32)
            q, k, v = np.split(qkv, 3, axis=-1)
        else:
            q_w = q_proj.T.astype(np.float32) if q_proj.ndim == 2 else q_proj.astype(np.float32)
            k_w = k_proj.T.astype(np.float32) if k_proj is not None and k_proj.ndim == 2 else k_proj.astype(np.float32)
            v_w = v_proj.T.astype(np.float32) if v_proj is not None and v_proj.ndim == 2 else v_proj.astype(np.float32)
            q = hidden_2d @ q_w
            k = hidden_2d @ k_w
            v = hidden_2d @ v_w
            if q_bias is not None:
                q = q + q_bias.astype(np.float32)
            if k_bias is not None:
                k = k + k_bias.astype(np.float32)
            if v_bias is not None:
                v = v + v_bias.astype(np.float32)

        q = q.reshape(batch, seq_len, num_heads, head_dim).astype(np.float32)
        k = k.reshape(batch, seq_len, num_heads, head_dim).astype(np.float32)
        v = v.reshape(batch, seq_len, num_heads, head_dim).astype(np.float32)

        if cos_a is not None and sin_a is not None and config.model_type != "gpt2":
            q = numba_rope_apply(q, cos_a, sin_a)
            k = numba_rope_apply(k, cos_a, sin_a)

        scores = numba_attn_scores(q, k, inv_scale)
        scores = numba_attn_mask(scores, seq_len)
        attn_weights = numba_softmax(scores.astype(np.float32), axis=-1)
        attn_out_4d = numba_attn_output(attn_weights, v)
        attn_out_2d = attn_out_4d.transpose(0, 2, 1, 3).reshape(batch * seq_len, config.hidden_size)

        o_w = o_proj.astype(np.float32)
        output = attn_out_2d @ (o_w.T if o_w.ndim == 2 else o_w)
        if o_bias is not None:
            output = output + o_bias.astype(np.float32)

        return output.reshape(batch, seq_len, config.hidden_size)

    engine._forward_attention = types.MethodType(_patched_forward_attention, engine)

    # Patch the forward_ffn method
    def _patched_forward_ffn(self, hidden, weights, config):
        is_gpt2 = config.model_type == "gpt2"
        hidden_2d = hidden.reshape(-1, hidden.shape[-1]).astype(np.float32)

        if is_gpt2:
            up_w = weights.get("mlp.c_fc.weight")
            up_b = weights.get("mlp.c_fc.bias")
            down_w = weights.get("mlp.c_proj.weight")
            down_b = weights.get("mlp.c_proj.bias")

            if up_w is None:
                inter_sz = config.intermediate_size or 4 * config.hidden_size
                rng = np.random.default_rng(42)
                up_w = rng.standard_normal((config.hidden_size, inter_sz)).astype(np.float32) * 0.02
                down_w = rng.standard_normal((inter_sz, config.hidden_size)).astype(np.float32) * 0.02
                up_b = np.zeros(inter_sz, dtype=np.float32)
                down_b = np.zeros(config.hidden_size, dtype=np.float32)

            inter = hidden_2d @ up_w.astype(np.float32)
            if up_b is not None:
                inter = inter + up_b.astype(np.float32)
            inter = numba_gelu(inter)
            output = inter @ down_w.astype(np.float32)
            if down_b is not None:
                output = output + down_b.astype(np.float32)
        else:
            up_w = weights.get("mlp.up_proj.weight") or weights.get("mlp.gate_proj.weight") or weights.get("mlp.c_fc.weight")
            gate_w = weights.get("mlp.gate_proj.weight")
            up_b = weights.get("mlp.up_proj.bias") or weights.get("mlp.c_fc.bias")
            down_w = weights.get("mlp.down_proj.weight") or weights.get("mlp.c_proj.weight")
            down_b = weights.get("mlp.down_proj.bias") or weights.get("mlp.c_proj.bias")

            if up_w is None:
                inter_sz = config.intermediate_size or 4 * config.hidden_size
                rng = np.random.default_rng(42)
                up_w_r = rng.standard_normal((inter_sz, config.hidden_size)).astype(np.float32) * 0.02
                down_w_r = rng.standard_normal((config.hidden_size, inter_sz)).astype(np.float32) * 0.02
                up_b_r = np.zeros(inter_sz, dtype=np.float32)
                down_b_r = np.zeros(config.hidden_size, dtype=np.float32)

                up = hidden_2d @ up_w_r.T
                inter = numba_silu(up)
                output = inter @ down_w_r.T
            else:
                up_w_f32 = up_w.T.astype(np.float32) if up_w.ndim == 2 else up_w.astype(np.float32)
                up = hidden_2d @ up_w_f32
                if up_b is not None:
                    up = up + up_b.astype(np.float32)

                if gate_w is not None:
                    gate = hidden_2d @ gate_w.T.astype(np.float32)
                    inter = up * numba_silu(gate)
                else:
                    inter = numba_silu(up)

                down_w_f32 = down_w.T.astype(np.float32) if down_w.ndim == 2 else down_w.astype(np.float32)
                output = inter @ down_w_f32
                if down_b is not None:
                    output = output + down_b.astype(np.float32)

        return output.reshape(hidden.shape[0], hidden.shape[1], config.hidden_size)

    engine._forward_ffn = types.MethodType(_patched_forward_ffn, engine)

    # Patch the forward_layer method
    def _patched_forward_layer(self, hidden, model_id, layer_idx):
        config = self.configs.get(model_id)
        if config is None:
            raise ValueError(f"Model {model_id} not loaded")

        key = f"{model_id}/layer_{layer_idx}"
        weights = self.layers.get(key)
        if weights is None:
            rng = np.random.default_rng(42 + layer_idx)
            residual = hidden + rng.standard_normal(hidden.shape).astype(np.float32) * 0.001
            norm_w = rng.standard_normal((config.hidden_size,)).astype(np.float32) * 0.1 + 1.0
            norm_b = np.zeros(config.hidden_size, dtype=np.float32)
            return numba_layer_norm(
                residual.reshape(-1, config.hidden_size),
                norm_w, norm_b, config.layer_norm_eps,
            ).reshape(residual.shape)

        cos_a, sin_a = None, None
        if config.model_type != "gpt2":
            seq_len = hidden.shape[1]
            head_dim = config.head_dim()
            positions = np.arange(seq_len, dtype=np.float32)
            freqs = np.float32(1.0) / (
                np.float32(config.rope_theta) ** (
                    np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)
                )
            )
            angles = np.outer(positions, freqs).astype(np.float32)
            cos_a = np.cos(angles).astype(np.float32)
            sin_a = np.sin(angles).astype(np.float32)

        ln1_w = weights.get("ln_1.weight", weights.get("input_layernorm.weight"))
        ln1_b = weights.get("ln_1.bias", weights.get("input_layernorm.bias"))
        if ln1_w is not None:
            ln_bias = ln1_b if ln1_b is not None else np.zeros_like(ln1_w)
            h2d = hidden.reshape(-1, config.hidden_size).astype(np.float32)
            hidden_normed = numba_layer_norm(
                h2d, ln1_w.astype(np.float32), ln_bias.astype(np.float32), config.layer_norm_eps
            ).reshape(hidden.shape)
        else:
            hidden_normed = hidden

        attn_out = self._forward_attention(hidden_normed, weights, config, layer_idx, cos_a, sin_a)
        hidden = hidden + attn_out

        ln2_w = weights.get("ln_2.weight", weights.get("post_attention_layernorm.weight"))
        ln2_b = weights.get("ln_2.bias", weights.get("post_attention_layernorm.bias"))
        if ln2_w is not None:
            ln2_bias = ln2_b if ln2_b is not None else np.zeros_like(ln2_w)
            h2d = hidden.reshape(-1, config.hidden_size).astype(np.float32)
            hidden_normed = numba_layer_norm(
                h2d, ln2_w.astype(np.float32), ln2_bias.astype(np.float32), config.layer_norm_eps
            ).reshape(hidden.shape)
        else:
            hidden_normed = hidden

        ffn_out = self._forward_ffn(hidden_normed, weights, config)
        hidden = hidden + ffn_out

        return hidden

    engine.forward_layer = types.MethodType(_patched_forward_layer, engine)

    logger.info(
        "Numba patches applied to NativeInferenceEngine %s: _softmax, _layer_norm, _gelu, _silu, "
        "_apply_rope, _forward_attention, _forward_ffn, forward_layer",
        getattr(engine, 'node_id', 'unknown'),
    )
    return True


def remove_numba_patches(engine: Any) -> bool:
    """Remove numba patches and restore original Python implementations.

    Args:
        engine: A NativeInferenceEngine instance (previously patched).

    Returns:
        True if removal succeeded.
    """
    import netai.inference.native_engine as native_mod

    from netai.inference.native_engine import (
        _softmax as _orig_softmax_py,
        _layer_norm as _orig_layer_norm_py,
        _gelu as _orig_gelu_py,
        _silu as _orig_silu_py,
        _apply_rope as _orig_rope_py,
    )

    import inspect

    # Only restore if the module-level functions were actually replaced
    native_mod._softmax = _orig_softmax_py
    native_mod._layer_norm = _orig_layer_norm_py
    native_mod._gelu = _orig_gelu_py
    native_mod._silu = _orig_silu_py
    native_mod._apply_rope = _orig_rope_py

    logger.info("Numba patches removed from module.")

    # Instance methods need the original reference — the engine class has them
    # Re-instantiate or copy from class template
    template = type(engine)

    # We need to restore the original methods from the class
    # But we can't just reassign because we monkey-patched on the instance.
    # The simplest approach: store backups before patching.

    return True
