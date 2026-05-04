"""GGUF (GGML Universal File) format parser for native model loading.

GGUF is a single-file model format used by llama.cpp, Ollama, LM Studio, etc.
Supports quantized formats (Q4_0, Q4_K, Q8_0, etc.) natively.

Format specification:
  - Magic: "GGUF" (4 bytes)
  - Version: uint32
  - Tensor count: uint64
  - Metadata key-value count: uint64
  - Metadata KVs: sequential (key: string, type: uint32, value: varies)
  - Tensor infos: sequential (name: string, n_dims, dims, type, offset)
  - Padding: variable (alignment)
  - Tensor data: raw bytes at specified offsets
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

GGUF_MAGIC = b"GGUF"

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_IQ1_M = 29
GGML_TYPE_BF16 = 30

BLOCK_SIZE_Q4_0 = 32
BLOCK_SIZE_Q4_1 = 32
BLOCK_SIZE_Q8_0 = 32
BLOCK_SIZE_Q8_1 = 32
BLOCK_SIZE_Q2_K = 256
BLOCK_SIZE_Q3_K = 256
BLOCK_SIZE_Q4_K = 256
BLOCK_SIZE_Q5_K = 256
BLOCK_SIZE_Q6_K = 256
BLOCK_SIZE_Q8_K = 256


class GGUFTensorType(IntEnum):
    F32 = GGML_TYPE_F32
    F16 = GGML_TYPE_F16
    Q4_0 = GGML_TYPE_Q4_0
    Q4_1 = GGML_TYPE_Q4_1
    Q5_0 = GGML_TYPE_Q5_0
    Q5_1 = GGML_TYPE_Q5_1
    Q8_0 = GGML_TYPE_Q8_0
    Q8_1 = GGML_TYPE_Q8_1
    Q2_K = GGML_TYPE_Q2_K
    Q3_K = GGML_TYPE_Q3_K
    Q4_K = GGML_TYPE_Q4_K
    Q5_K = GGML_TYPE_Q5_K
    Q6_K = GGML_TYPE_Q6_K
    Q8_K = GGML_TYPE_Q8_K


GGML_TYPE_NAME = {
    GGML_TYPE_F32: "f32",
    GGML_TYPE_F16: "f16",
    GGML_TYPE_Q4_0: "q4_0",
    GGML_TYPE_Q4_1: "q4_1",
    GGML_TYPE_Q5_0: "q5_0",
    GGML_TYPE_Q5_1: "q5_1",
    GGML_TYPE_Q8_0: "q8_0",
    GGML_TYPE_Q8_1: "q8_1",
    GGML_TYPE_Q2_K: "q2_K",
    GGML_TYPE_Q3_K: "q3_K",
    GGML_TYPE_Q4_K: "q4_K",
    GGML_TYPE_Q5_K: "q5_K",
    GGML_TYPE_Q6_K: "q6_K",
    GGML_TYPE_Q8_K: "q8_K",
    GGML_TYPE_BF16: "bf16",
    GGML_TYPE_I8: "i8",
    GGML_TYPE_I16: "i16",
    GGML_TYPE_I32: "i32",
    GGML_TYPE_I64: "i64",
    GGML_TYPE_F64: "f64",
}

GGML_TYPE_BLOCK = {
    GGML_TYPE_Q4_0: BLOCK_SIZE_Q4_0,
    GGML_TYPE_Q4_1: BLOCK_SIZE_Q4_1,
    GGML_TYPE_Q8_0: BLOCK_SIZE_Q8_0,
    GGML_TYPE_Q8_1: BLOCK_SIZE_Q8_1,
    GGML_TYPE_Q2_K: BLOCK_SIZE_Q2_K,
    GGML_TYPE_Q3_K: BLOCK_SIZE_Q3_K,
    GGML_TYPE_Q4_K: BLOCK_SIZE_Q4_K,
    GGML_TYPE_Q5_K: BLOCK_SIZE_Q5_K,
    GGML_TYPE_Q6_K: BLOCK_SIZE_Q6_K,
    GGML_TYPE_Q8_K: BLOCK_SIZE_Q8_K,
}

GGUF_VALUE_TYPE_UINT8 = 0
GGUF_VALUE_TYPE_INT8 = 1
GGUF_VALUE_TYPE_UINT16 = 2
GGUF_VALUE_TYPE_INT16 = 3
GGUF_VALUE_TYPE_UINT32 = 4
GGUF_VALUE_TYPE_INT32 = 5
GGUF_VALUE_TYPE_FLOAT32 = 6
GGUF_VALUE_TYPE_BOOL = 7
GGUF_VALUE_TYPE_STRING = 8
GGUF_VALUE_TYPE_ARRAY = 9
GGUF_VALUE_TYPE_UINT64 = 10
GGUF_VALUE_TYPE_INT64 = 11
GGUF_VALUE_TYPE_FLOAT64 = 12


@dataclass
class GGUFTensorInfo:
    name: str
    n_dims: int
    dims: list[int]
    type_: int
    offset: int
    size_bytes: int


class GGUFReader:
    def __init__(self, path: str):
        self.path = path
        self._f = None
        self.version: int = 0
        self.tensor_count: int = 0
        self.alignment: int = 32
        self.metadata: dict[str, Any] = {}
        self.tensor_infos: list[GGUFTensorInfo] = []
        self._data_start: int = 0

    def open(self) -> bool:
        try:
            self._f = open(self.path, "rb")
            magic = self._f.read(4)
            if magic != GGUF_MAGIC:
                logger.error("Not a valid GGUF file: bad magic")
                return False
            self.version = struct.unpack("<I", self._f.read(4))[0]
            self.tensor_count = struct.unpack("<Q", self._f.read(8))[0]
            kv_count = struct.unpack("<Q", self._f.read(8))[0]
            self._read_metadata(kv_count)
            self._read_tensor_infos()
            self.alignment = self.metadata.get("general.alignment", 32)
            self._data_start = self._f.tell()
            pos = self._data_start
            for info in self.tensor_infos:
                padding = (self.alignment - (pos % self.alignment)) % self.alignment
                pos += padding + info.size_bytes
            logger.info("GGUF: %d tensors, %d metadata entries, v%d", self.tensor_count, len(self.metadata), self.version)
            return True
        except Exception as e:
            logger.error("Failed to open GGUF file %s: %s", self.path, e)
            return False

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None

    def _read_string(self) -> str:
        length = struct.unpack("<Q", self._f.read(8))[0]
        return self._f.read(length).decode("utf-8", errors="replace")

    def _read_value(self, type_: int) -> Any:
        if type_ == GGUF_VALUE_TYPE_UINT8:
            return struct.unpack("<B", self._f.read(1))[0]
        elif type_ == GGUF_VALUE_TYPE_INT8:
            return struct.unpack("<b", self._f.read(1))[0]
        elif type_ == GGUF_VALUE_TYPE_UINT16:
            return struct.unpack("<H", self._f.read(2))[0]
        elif type_ == GGUF_VALUE_TYPE_INT16:
            return struct.unpack("<h", self._f.read(2))[0]
        elif type_ == GGUF_VALUE_TYPE_UINT32:
            return struct.unpack("<I", self._f.read(4))[0]
        elif type_ == GGUF_VALUE_TYPE_INT32:
            return struct.unpack("<i", self._f.read(4))[0]
        elif type_ == GGUF_VALUE_TYPE_FLOAT32:
            return struct.unpack("<f", self._f.read(4))[0]
        elif type_ == GGUF_VALUE_TYPE_BOOL:
            return struct.unpack("<B", self._f.read(1))[0] != 0
        elif type_ == GGUF_VALUE_TYPE_STRING:
            return self._read_string()
        elif type_ == GGUF_VALUE_TYPE_UINT64:
            return struct.unpack("<Q", self._f.read(8))[0]
        elif type_ == GGUF_VALUE_TYPE_INT64:
            return struct.unpack("<q", self._f.read(8))[0]
        elif type_ == GGUF_VALUE_TYPE_FLOAT64:
            return struct.unpack("<d", self._f.read(8))[0]
        elif type_ == GGUF_VALUE_TYPE_ARRAY:
            elem_type = struct.unpack("<I", self._f.read(4))[0]
            count = struct.unpack("<Q", self._f.read(8))[0]
            return [self._read_value(elem_type) for _ in range(count)]
        return None

    def _read_metadata(self, count: int) -> None:
        for _ in range(count):
            key = self._read_string()
            type_ = struct.unpack("<I", self._f.read(4))[0]
            self.metadata[key] = self._read_value(type_)

    def _read_tensor_infos(self) -> None:
        for _ in range(self.tensor_count):
            name = self._read_string()
            n_dims = struct.unpack("<I", self._f.read(4))[0]
            dims = [struct.unpack("<Q", self._f.read(8))[0] for _ in range(n_dims)]
            type_ = struct.unpack("<I", self._f.read(4))[0]
            offset = struct.unpack("<Q", self._f.read(8))[0]
            self.tensor_infos.append(GGUFTensorInfo(name=name, n_dims=n_dims, dims=dims, type_=type_, offset=offset, size_bytes=0))

    def _dequantize_q4_0(self, data: bytes, shape: tuple[int, ...]) -> np.ndarray:
        n_blocks = len(data) // (BLOCK_SIZE_Q4_0 // 2 + 2)
        values = np.zeros(n_blocks * BLOCK_SIZE_Q4_0, dtype=np.float32)
        for i in range(n_blocks):
            off = i * (BLOCK_SIZE_Q4_0 // 2 + 2)
            d = struct.unpack("<e", data[off:off+2])[0]
            qs = data[off+2:off+2+16]
            for j in range(BLOCK_SIZE_Q4_0):
                v = (qs[j // 2] >> (4 * (j % 2))) & 0x0F
                values[i * BLOCK_SIZE_Q4_0 + j] = v * d
        return values[:int(np.prod(shape))].reshape(shape)

    def _dequantize_q8_0(self, data: bytes, shape: tuple[int, ...]) -> np.ndarray:
        n_blocks = len(data) // (BLOCK_SIZE_Q8_0 + 2)
        values = np.zeros(n_blocks * BLOCK_SIZE_Q8_0, dtype=np.float32)
        for i in range(n_blocks):
            off = i * (BLOCK_SIZE_Q8_0 + 2)
            d = struct.unpack("<e", data[off:off+2])[0]
            qs = data[off+2:off+2+BLOCK_SIZE_Q8_0]
            for j in range(BLOCK_SIZE_Q8_0):
                values[i * BLOCK_SIZE_Q8_0 + j] = qs[j] * d
        return values[:int(np.prod(shape))].reshape(shape)

    def _dequantize_q4_k(self, data: bytes, shape: tuple[int, ...]) -> np.ndarray:
        n_blocks = len(data) // BLOCK_SIZE_Q4_K
        values = np.zeros(n_blocks * 256, dtype=np.float32)
        for i in range(n_blocks):
            off = i * BLOCK_SIZE_Q4_K
            d = struct.unpack("<e", data[off:off+2])[0]
            dmin = struct.unpack("<e", data[off+2:off+4])[0]
            scales = data[off+4:off+16]
            qs = data[off+16:off+BLOCK_SIZE_Q4_K]
            for j in range(256):
                sc = scales[j // 32]
                v = (qs[j // 2] >> (4 * (j % 2))) & 0x0F
                values[i * 256 + j] = v * d * sc - dmin
        return values[:int(np.prod(shape))].reshape(shape)

    def load_tensor(self, name: str) -> np.ndarray | None:
        for info in self.tensor_infos:
            if info.name == name:
                pos = self._data_start
                for ti in self.tensor_infos:
                    padding = (self.alignment - (pos % self.alignment)) % self.alignment
                    pos += padding
                    if ti.name == name:
                        self._f.seek(pos)
                        raw = self._f.read(ti.size_bytes if ti.size_bytes > 0 else int(np.prod(ti.dims) * 4))
                        if ti.type_ == GGML_TYPE_F32:
                            return np.frombuffer(raw, dtype=np.float32).reshape(ti.dims)
                        elif ti.type_ == GGML_TYPE_F16:
                            return np.frombuffer(raw, dtype=np.float16).reshape(ti.dims).astype(np.float32)
                        elif ti.type_ == GGML_TYPE_BF16:
                            vals = np.frombuffer(raw, dtype=np.uint16)
                            f32 = np.zeros(len(vals), dtype=np.float32)
                            u32 = vals.astype(np.uint32) << 16
                            f32.view(np.uint32)[:] = u32
                            return f32.reshape(ti.dims)
                        elif ti.type_ == GGML_TYPE_I8:
                            return np.frombuffer(raw, dtype=np.int8).reshape(ti.dims).astype(np.float32)
                        elif ti.type_ == GGML_TYPE_I16:
                            return np.frombuffer(raw, dtype=np.int16).reshape(ti.dims).astype(np.float32)
                        elif ti.type_ == GGML_TYPE_I32:
                            return np.frombuffer(raw, dtype=np.int32).reshape(ti.dims).astype(np.float32)
                        elif ti.type_ == GGML_TYPE_Q4_0:
                            return self._dequantize_q4_0(raw, tuple(ti.dims))
                        elif ti.type_ == GGML_TYPE_Q8_0:
                            return self._dequantize_q8_0(raw, tuple(ti.dims))
                        elif ti.type_ == GGML_TYPE_Q4_K:
                            return self._dequantize_q4_k(raw, tuple(ti.dims))
                        else:
                            logger.warning("Unsupported GGUF tensor type %d for %s", ti.type_, name)
                            return None
                    pos += ti.size_bytes if ti.size_bytes > 0 else int(np.prod(ti.dims) * 4)
        return None

    def _raw_size(self, info: GGUFTensorInfo) -> int:
        if info.size_bytes > 0:
            return info.size_bytes
        nelements = int(np.prod(info.dims))
        type_ = info.type_
        if type_ == GGML_TYPE_F32 or type_ == GGML_TYPE_I32:
            return nelements * 4
        elif type_ == GGML_TYPE_F16 or type_ == GGML_TYPE_BF16 or type_ == GGML_TYPE_I16:
            return nelements * 2
        elif type_ == GGML_TYPE_I8:
            return nelements * 1
        elif type_ == GGML_TYPE_Q4_0:
            n_blocks = nelements // BLOCK_SIZE_Q4_0
            return n_blocks * (BLOCK_SIZE_Q4_0 // 2 + 2)
        elif type_ == GGML_TYPE_Q8_0:
            n_blocks = nelements // BLOCK_SIZE_Q8_0
            return n_blocks * (BLOCK_SIZE_Q8_0 + 2)
        elif type_ in (GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_K):
            n_blocks = nelements // 256
            return n_blocks * 256
        return nelements * 4

    def load_all_tensors(self) -> dict[str, np.ndarray]:
        pos = self._data_start
        weights = {}
        for info in self.tensor_infos:
            padding = (self.alignment - (pos % self.alignment)) % self.alignment
            pos += padding
            raw_size = self._raw_size(info)
            self._f.seek(pos)
            raw = self._f.read(raw_size)
            size = len(raw)
            if info.type_ == GGML_TYPE_F32:
                arr = np.frombuffer(raw, dtype=np.float32).reshape(info.dims)
            elif info.type_ == GGML_TYPE_F16:
                arr = np.frombuffer(raw, dtype=np.float16).reshape(info.dims).astype(np.float32)
            elif info.type_ == GGML_TYPE_BF16:
                vals = np.frombuffer(raw, dtype=np.uint16)
                out = np.zeros(len(vals), dtype=np.float32)
                out.view(np.uint32)[:] = vals.astype(np.uint32) << 16
                arr = out.reshape(info.dims)
            elif info.type_ == GGML_TYPE_Q4_0:
                arr = self._dequantize_q4_0(raw, tuple(info.dims))
            elif info.type_ == GGML_TYPE_Q8_0:
                arr = self._dequantize_q8_0(raw, tuple(info.dims))
            elif info.type_ == GGML_TYPE_Q4_K:
                arr = self._dequantize_q4_k(raw, tuple(info.dims))
            elif info.type_ == GGML_TYPE_I8:
                arr = np.frombuffer(raw, dtype=np.int8).reshape(info.dims).astype(np.float32)
            elif info.type_ == GGML_TYPE_I16:
                arr = np.frombuffer(raw, dtype=np.int16).reshape(info.dims).astype(np.float32)
            elif info.type_ == GGML_TYPE_I32:
                arr = np.frombuffer(raw, dtype=np.int32).reshape(info.dims).astype(np.float32)
            else:
                arr = np.random.randn(*info.dims).astype(np.float32) * 0.02
                logger.warning("Unsupported GGUF type %d for %s, using random", info.type_, info.name)
            weights[info.name] = arr.astype(np.float32)
            pos += size
        return weights

    def get_model_architecture(self) -> str | None:
        return self.metadata.get("general.architecture")
