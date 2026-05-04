"""Activation compression for pipeline-parallel inter-node communication.

Implements 8-bit quantization (dynamic min-max) for hidden state activations
sent between pipeline stages. Based on Petals' approach: quantize to INT8,
transmit quantized values + metadata, dequantize at receiver. Optionally
supports FP16 sparse residual for high-quality large-model inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuantizedTensor:
    """Quantized tensor ready for network transmission."""
    data: bytes
    shape: list[int]
    dtype_original: str
    scale: float
    zero_point: float
    compression_ratio: float


class ActivationCompressor:
    """Compresses/decompresses activation tensors for pipeline transfer."""

    QUANT_BITS = 8
    QUANT_MAX = 2 ** QUANT_BITS - 1

    def __init__(self, bits: int = 8, use_residual: bool = False):
        self._bits = max(4, min(bits, 8))
        self._max_val = 2 ** self._bits - 1
        self._use_residual = use_residual
        self._compressed_bytes_total = 0
        self._uncompressed_bytes_total = 0

    def compress(self, tensor: np.ndarray) -> QuantizedTensor:
        """Compress a float32 activation to INT8 (or lower bits)."""
        original_bytes = tensor.nbytes
        t_min = tensor.min()
        t_max = tensor.max()
        rng = t_max - t_min
        if abs(rng) < 1e-6:
            scale = 1.0
            zero_point = t_min
            quantized = np.zeros(tensor.shape, dtype=np.uint8)
        else:
            scale = rng / self._max_val
            zero_point = t_min
            quantized = np.clip(
                ((tensor - t_min) / scale + 0.5).astype(np.int32),
                0, self._max_val,
            ).astype(np.uint8)

        if self._bits < 8:
            quantized = quantized >> (8 - self._bits)

        data = self._pack_bits(quantized) if self._bits < 8 else quantized.tobytes()
        compressed_bytes = len(data) + 4 + 4
        ratio = original_bytes / max(compressed_bytes, 1)
        self._compressed_bytes_total += compressed_bytes
        self._uncompressed_bytes_total += original_bytes

        return QuantizedTensor(
            data=data,
            shape=list(tensor.shape),
            dtype_original=str(tensor.dtype),
            scale=scale,
            zero_point=zero_point,
            compression_ratio=ratio,
        )

    def _pack_bits(self, arr: np.ndarray) -> bytes:
        values_per_byte = 8 // self._bits
        flat = arr.flatten().astype(np.uint32)
        padded = np.zeros((len(flat) + 1) // 2 * 2, dtype=np.uint32)
        padded[:len(flat)] = flat
        packed = np.zeros(len(padded) // values_per_byte, dtype=np.uint8)
        for i in range(values_per_byte):
            packed |= (padded[i::values_per_byte][:len(packed)] & self._max_val).astype(np.uint8) << (i * self._bits)
        return packed.tobytes()

    def _unpack_bits(self, data: bytes, shape: tuple[int, ...]) -> np.ndarray:
        values_per_byte = 8 // self._bits
        packed = np.frombuffer(data, dtype=np.uint8)
        total = int(np.prod(shape))
        result = np.zeros(total, dtype=np.uint8)
        mask = self._max_val
        for i in range(values_per_byte):
            idx = i * self._bits
            vals = (packed[:len(result) // values_per_byte] >> idx) & mask
            if i < len(result):
                result[i::values_per_byte] = vals[:len(result[i::values_per_byte])]
        return result[:total].reshape(shape)

    def decompress(self, q: QuantizedTensor) -> np.ndarray:
        """Decompress a QuantizedTensor back to float32."""
        if self._bits < 8:
            quantized = self._unpack_bits(q.data, tuple(q.shape))
        else:
            quantized = np.frombuffer(q.data, dtype=np.uint8).reshape(q.shape)
        return quantized.astype(np.float32) * q.scale + q.zero_point

    def compress_residual(self, tensor: np.ndarray) -> dict:
        """Compress with 8-bit quantization + sparse FP16 residual."""
        q = self.compress(tensor)
        reconstructed = self.decompress(q)
        residual = tensor - reconstructed
        abs_res = np.abs(residual)
        threshold = np.sort(abs_res.flatten())[-max(1, int(residual.size * 0.01))]
        sparse_indices = np.where(abs_res >= threshold)
        sparse_values = residual[sparse_indices].astype(np.float16)

        return {
            "shape": q.shape,
            "dtype": q.dtype_original,
            "data_hex": q.data.hex(),
            "scale": q.scale,
            "zero_point": q.zero_point,
            "residual_indices": [list(ax) for ax in sparse_indices],
            "residual_values_hex": sparse_values.tobytes().hex(),
            "compression_ratio": q.compression_ratio,
        }

    def decompress_residual(self, compressed: dict) -> np.ndarray:
        """Decompress with residual correction."""
        data_hex = compressed.get("data_hex")
        shape = compressed.get("shape", [])
        dtype = compressed.get("dtype", "float32")
        scale = compressed.get("scale", 1.0)
        zero_point = compressed.get("zero_point", 0.0)
        rv_hex = compressed.get("residual_values_hex")
        rv_indices = compressed.get("residual_indices")

        q = QuantizedTensor(
            data=bytes.fromhex(data_hex) if data_hex else b"",
            shape=shape, dtype_original=dtype,
            scale=scale, zero_point=zero_point,
            compression_ratio=compressed.get("compression_ratio", 1.0),
        )
        result = self.decompress(q)

        if rv_hex and rv_indices and len(rv_indices) >= 2:
            idx_tuple = tuple(np.array(ax, dtype=np.int64) for ax in rv_indices)
            values = np.frombuffer(bytes.fromhex(rv_hex), dtype=np.float16).astype(np.float32)
            if len(idx_tuple[0]) == len(values):
                result[idx_tuple] += values

        return result

    def get_stats(self) -> dict:
        return {
            "compressed_bytes_total": self._compressed_bytes_total,
            "uncompressed_bytes_total": self._uncompressed_bytes_total,
            "overall_ratio": self._uncompressed_bytes_total / max(self._compressed_bytes_total, 1),
        }


def quantize_activation(activation: np.ndarray, bits: int = 8) -> dict:
    """Quick one-shot quantization for API integration."""
    comp = ActivationCompressor(bits=bits)
    q = comp.compress(activation)
    return {
        "data_hex": q.data.hex(),
        "shape": q.shape,
        "dtype": q.dtype_original,
        "scale": q.scale,
        "zero_point": q.zero_point,
    }


def dequantize_activation(compressed: dict) -> np.ndarray:
    """Quick one-shot dequantization for API integration."""
    return ActivationCompressor().decompress(QuantizedTensor(
        data=bytes.fromhex(compressed["data_hex"]),
        shape=compressed["shape"],
        dtype_original=compressed["dtype"],
        scale=compressed["scale"],
        zero_point=compressed["zero_point"],
        compression_ratio=1.0,
    ))
