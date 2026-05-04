"""Tests for GGUF model loading integration with NativeInferenceEngine."""

import json
import os
import struct
import tempfile
import time

import numpy as np
import pytest

from netai.inference.native_engine import NativeInferenceEngine, TransformerConfig
from netai.inference.gguf_parser import (
    GGUFReader,
    GGUF_MAGIC,
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q8_0,
    GGML_TYPE_BF16,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
)
from netai.inference.native_engine import (
    _softmax,
    _layer_norm,
    _gelu,
    _silu,
)

GGUF_VALUE_TYPE_UINT32 = 4
GGUF_VALUE_TYPE_FLOAT32 = 6
GGUF_VALUE_TYPE_STRING = 8
GGUF_VALUE_TYPE_ARRAY = 9
GGUF_VALUE_TYPE_UINT64 = 10
GGUF_VALUE_TYPE_INT64 = 11
GGUF_VALUE_TYPE_FLOAT64 = 12


def _write_gguf_string(f, s):
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def _write_gguf_value(f, type_, value):
    if type_ == GGUF_VALUE_TYPE_UINT32:
        f.write(struct.pack("<I", value))
    elif type_ == GGUF_VALUE_TYPE_UINT64:
        f.write(struct.pack("<Q", value))
    elif type_ == GGUF_VALUE_TYPE_INT64:
        f.write(struct.pack("<q", value))
    elif type_ == GGUF_VALUE_TYPE_FLOAT32:
        f.write(struct.pack("<f", value))
    elif type_ == GGUF_VALUE_TYPE_FLOAT64:
        f.write(struct.pack("<d", value))
    elif type_ == GGUF_VALUE_TYPE_STRING:
        _write_gguf_string(f, value)
    elif type_ == GGUF_VALUE_TYPE_ARRAY:
        elem_type = type(value[0]) if isinstance(value, list) and len(value) > 0 else GGUF_VALUE_TYPE_UINT32
        if isinstance(value[0], str):
            elem_type = GGUF_VALUE_TYPE_STRING
        elif isinstance(value[0], int):
            elem_type = GGUF_VALUE_TYPE_INT64 if value[0] < 0 else GGUF_VALUE_TYPE_UINT64
        elif isinstance(value[0], float):
            elem_type = GGUF_VALUE_TYPE_FLOAT64
        f.write(struct.pack("<I", elem_type))
        f.write(struct.pack("<Q", len(value)))
        for elem in value:
            _write_gguf_value(f, elem_type, elem)


def _write_gguf_kv(f, key, type_, value):
    _write_gguf_string(f, key)
    f.write(struct.pack("<I", type_))
    _write_gguf_value(f, type_, value)


def _build_gguf_metadata(config, tensor_names, tensor_shapes, tensor_types):
    H = config["hidden_size"]
    L = config["num_layers"]
    V = config["vocab_size"]
    I = config["intermediate_size"]

    kvs = []
    kv_dict = {
        "general.architecture": "llama",
        "llama.vocab_length": V,
        "llama.block_count": L,
        "llama.embedding_length": H,
        "llama.feed_forward_length": I,
        "llama.attention.head_count": config["num_heads"],
        "llama.context_length": config["max_position_embeddings"],
        "llama.attention.layer_norm_rms_epsilon": config.get("layer_norm_eps", 1e-5),
        "llama.rope.theta": config.get("rope_theta", 10000.0),
    }
    for k, v in kv_dict.items():
        kvs.append((k, v))

    return kvs


def create_synthetic_gguf(
    path, config=None, tensor_names=None, tensor_shapes=None,
    tensor_data=None, tensor_types=None, bad_magic=False,
    truncated=False, extra_metadata=None,
):
    if config is None:
        config = {
            "hidden_size": 32,
            "num_layers": 2,
            "vocab_size": 100,
            "intermediate_size": 64,
            "num_heads": 4,
            "max_position_embeddings": 64,
            "layer_norm_eps": 1e-5,
            "rope_theta": 10000.0,
        }

    H = config["hidden_size"]
    L = config["num_layers"]
    V = config["vocab_size"]
    I = config["intermediate_size"]
    N = config["num_heads"]

    if tensor_data is None:
        rng = np.random.default_rng(42)
        tensor_names = []
        tensor_shapes = {}
        tensor_types = {}
        tensor_data = {}

        tensor_names.append("token_embd.weight")
        tensor_shapes["token_embd.weight"] = (V, H)
        tensor_types["token_embd.weight"] = GGML_TYPE_F32
        tensor_data["token_embd.weight"] = rng.standard_normal((V, H)).astype(np.float32) * 0.02

        for l in range(L):
            prefix = f"blk.{l}."
            for suf, shape in [
                ("attn_q.weight", (H, H)),
                ("attn_k.weight", (H, H)),
                ("attn_v.weight", (H, H)),
                ("attn_output.weight", (H, H)),
                ("attn_norm.weight", (H,)),
                ("ffn_gate.weight", (I, H)),
                ("ffn_up.weight", (I, H)),
                ("ffn_down.weight", (H, I)),
                ("ffn_norm.weight", (H,)),
            ]:
                name = prefix + suf
                tensor_names.append(name)
                tensor_shapes[name] = shape
                tensor_types[name] = GGML_TYPE_F32
                tensor_data[name] = rng.standard_normal(shape).astype(np.float32) * 0.02

        tensor_names.append("output_norm.weight")
        tensor_shapes["output_norm.weight"] = (H,)
        tensor_types["output_norm.weight"] = GGML_TYPE_F32
        tensor_data["output_norm.weight"] = rng.standard_normal((H,)).astype(np.float32) * 0.02

    with open(path, "wb") as f:
        f.write(b"XXXX" if bad_magic else GGUF_MAGIC)
        f.write(struct.pack("<I", 3))
        metadata_entries = _build_gguf_metadata(config, tensor_names, tensor_shapes, None)
        if extra_metadata:
            metadata_entries.extend(extra_metadata)
        f.write(struct.pack("<Q", len(tensor_names)))
        f.write(struct.pack("<Q", len(metadata_entries)))

        for key, value in metadata_entries:
            if isinstance(value, str):
                _write_gguf_kv(f, key, GGUF_VALUE_TYPE_STRING, value)
            elif isinstance(value, int):
                _write_gguf_kv(f, key, GGUF_VALUE_TYPE_UINT32, value)
            elif isinstance(value, float):
                _write_gguf_kv(f, key, GGUF_VALUE_TYPE_FLOAT32, value)
            elif isinstance(value, list):
                _write_gguf_kv(f, key, GGUF_VALUE_TYPE_ARRAY, value)

        total_raw = 0
        for name in tensor_names:
            _write_gguf_string(f, name)
            shape = tensor_shapes[name]
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", tensor_types[name]))
            f.write(struct.pack("<Q", 0))
            total_raw += int(np.prod(shape)) * 4

        alignment = 32
        padding = (alignment - (f.tell() % alignment)) % alignment
        f.write(b'\x00' * padding)

        data_start = f.tell()
        offset = data_start
        for name in tensor_names:
            padding = (alignment - (offset % alignment)) % alignment
            offset += padding
        del offset

        if not truncated:
            pos = f.tell()
            for name in tensor_names:
                padding = (alignment - (pos % alignment)) % alignment
                pos += padding
                f.seek(pos)
                raw = tensor_data[name].tobytes()
                f.write(raw)
                pos += len(raw)
            f.seek(pos)
        else:
            half_point = total_raw // 3
            f.write(b'\x00' * half_point)

    return path


def _make_quantized_tensor(rng, shape, q_type, seed=42):
    if q_type == GGML_TYPE_F32:
        return rng.standard_normal(shape, dtype=np.float32) * 0.02
    base = rng.standard_normal(shape, dtype=np.float32) * 0.02
    if q_type == GGML_TYPE_Q4_0:
        block_size = 32
        total = int(np.prod(shape))
        data = bytearray()
        for i in range(0, total, block_size):
            block_data = base.ravel()[i:i+block_size].astype(np.float32)
            max_val = max(abs(block_data.min()), abs(block_data.max()))
            max_val = max(max_val, 1e-8)
            d = max_val / 7.5
            packed = np.zeros(block_size, dtype=np.float32)
            for j in range(min(block_size, total - i)):
                q = max(0, min(15, int(round(((block_data[j] / d) + 8)))))
                packed[j] = (q - 8) * d
                if j % 2 == 0:
                    data.append(q & 0x0F)
                else:
                    data[-1] |= (q & 0x0F) << 4
            if block_size > (total - i):
                rem = block_size - (total - i)
                for j in range(rem // 2):
                    data.append(0)
            d_f16 = struct.pack("<e", np.float16(d).view(np.uint16)[0])
            data[len(data) - (block_size // 2) - 2:len(data) - (block_size // 2)] = d_f16
            data = data[:-(block_size // 2)] + d_f16 + data[-(block_size // 2):]
        values = np.zeros_like(base).ravel()
        for i in range(0, total, block_size):
            off = i * (block_size // 2 + 2) // block_size
            d_raw = bytes(data[off:off+2])
            d = struct.unpack("<e", d_raw)[0]
            for j in range(min(block_size, total - i)):
                q = data[off + 2 + j // 2]
                if j % 2 == 0:
                    q &= 0x0F
                else:
                    q >>= 4
                values[i + j] = (q - 8) * d
        return values.reshape(shape).astype(np.float32)
    return base


class TestGGUFIntegration:
    def test_create_synthetic_gguf_load_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "model.gguf")
            config = {
                "hidden_size": 64,
                "num_layers": 3,
                "vocab_size": 200,
                "intermediate_size": 128,
                "num_heads": 8,
                "max_position_embeddings": 128,
                "layer_norm_eps": 1e-6,
                "rope_theta": 50000.0,
            }
            create_synthetic_gguf(gguf_path, config=config)

            engine = NativeInferenceEngine(node_id="gguf-config-test")
            shard = engine.load_gguf_model("test-gguf-config", gguf_path)
            assert shard.loaded
            assert shard.num_layers == 3

            loaded_config = engine.configs["test-gguf-config"]
            assert loaded_config.hidden_size == 64
            assert loaded_config.num_layers == 3
            assert loaded_config.num_heads == 8
            assert loaded_config.vocab_size == 200
            assert loaded_config.max_position_embeddings == 128
            assert abs(loaded_config.layer_norm_eps - 1e-6) < 1e-12
            assert abs(loaded_config.rope_theta - 50000.0) < 1e-6

    def test_load_gguf_model_nonexistent_file(self):
        engine = NativeInferenceEngine(node_id="no-file-test")
        shard = engine.load_gguf_model("missing-model", "/tmp/no_such_file.gguf")
        assert not shard.loaded

    def test_load_gguf_model_bad_magic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "bad.gguf")
            create_synthetic_gguf(gguf_path, bad_magic=True)
            engine = NativeInferenceEngine(node_id="bad-magic-test")
            shard = engine.load_gguf_model("bad-magic", gguf_path)
            assert not shard.loaded

    def test_load_gguf_model_f32_forward_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "f32_model.gguf")
            config = {
                "hidden_size": 32,
                "num_layers": 2,
                "vocab_size": 100,
                "intermediate_size": 64,
                "num_heads": 4,
                "max_position_embeddings": 64,
                "layer_norm_eps": 1e-5,
                "rope_theta": 10000.0,
            }
            create_synthetic_gguf(gguf_path, config=config)

            engine = NativeInferenceEngine(node_id="f32-fwd-test")
            shard = engine.load_gguf_model("f32-model", gguf_path)
            assert shard.loaded
            assert shard.num_layers == 2

            hidden = np.random.randn(1, 8, 32).astype(np.float32)
            out = engine.forward_layer(hidden, "f32-model", 0)
            assert out.shape == hidden.shape
            out = engine.forward_layer(out, "f32-model", 1)
            assert out.shape == hidden.shape

    def test_load_gguf_model_full_generate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "gen_model.gguf")
            config = {
                "hidden_size": 16,
                "num_layers": 2,
                "vocab_size": 50,
                "intermediate_size": 32,
                "num_heads": 2,
                "max_position_embeddings": 32,
                "layer_norm_eps": 1e-5,
                "rope_theta": 10000.0,
            }
            create_synthetic_gguf(gguf_path, config=config)

            engine = NativeInferenceEngine(node_id="gen-test")
            shard = engine.load_gguf_model("gen-model", gguf_path)
            assert shard.loaded

            result = engine.generate(
                model_id="gen-model",
                prompt_tokens=[1, 2, 3],
                max_tokens=8,
                temperature=0.7,
            )
            assert "tokens" in result
            assert result["num_generated"] == 8
            assert len(result["tokens"]) == 3 + 8
            assert result["latency_ms"] > 0
            assert result["tokens_per_second"] > 0

            for tk in result["generated_tokens"]:
                assert 0 <= tk < 50

    def test_load_gguf_model_with_kv_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "kvc_model.gguf")
            config = {
                "hidden_size": 16,
                "num_layers": 2,
                "vocab_size": 50,
                "intermediate_size": 32,
                "num_heads": 2,
                "max_position_embeddings": 32,
                "layer_norm_eps": 1e-5,
                "rope_theta": 10000.0,
            }
            create_synthetic_gguf(gguf_path, config=config)

            engine = NativeInferenceEngine(node_id="kvc-test")
            engine.load_gguf_model("kvc-model", gguf_path)

            tokens = [1, 2, 3]
            input_ids = np.array(tokens, dtype=np.int64).reshape(1, -1)
            hidden = engine.embed_tokens["kvc-model"][input_ids]

            kv_caches = {}
            for i in range(2):
                hidden, new_kv = engine.forward_layer_cached(hidden, "kvc-model", i, None)
                kv_caches[i] = new_kv

            assert hidden.shape[-1] == 16
            assert len(kv_caches) == 2
            for kv in kv_caches.values():
                assert len(kv) == 2

    def test_load_gguf_via_load_model_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "model.gguf")
            config = {
                "hidden_size": 32,
                "num_layers": 2,
                "vocab_size": 100,
                "intermediate_size": 64,
                "num_heads": 4,
                "max_position_embeddings": 64,
                "layer_norm_eps": 1e-5,
                "rope_theta": 10000.0,
            }
            create_synthetic_gguf(gguf_path, config=config)

            engine = NativeInferenceEngine(node_id="dir-test")
            shard = engine.load_model("dir-model", tmpdir)
            assert shard.loaded
            assert shard.num_layers == 2

    def test_load_gguf_via_load_model_direct_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "model.gguf")
            create_synthetic_gguf(gguf_path)

            engine = NativeInferenceEngine(node_id="direct-test")
            shard = engine.load_model("direct-model", gguf_path)
            assert shard.loaded
            assert shard.num_layers == 2

    def test_truncated_gguf_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "truncated.gguf")
            config = {
                "hidden_size": 32,
                "num_layers": 2,
                "vocab_size": 100,
                "intermediate_size": 64,
                "num_heads": 4,
                "max_position_embeddings": 64,
                "layer_norm_eps": 1e-5,
                "rope_theta": 10000.0,
            }
            create_synthetic_gguf(gguf_path, config=config, truncated=True)

            engine = NativeInferenceEngine(node_id="trunc-test")
            shard = engine.load_gguf_model("trunc-model", gguf_path)
            assert not shard.loaded

    def test_load_gguf_model_output_proj_and_norm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "out_model.gguf")
            rng = np.random.default_rng(42)
            V, H, I = 80, 24, 48
            config = {
                "hidden_size": H,
                "num_layers": 2,
                "vocab_size": V,
                "intermediate_size": I,
                "num_heads": 4,
                "max_position_embeddings": 32,
                "layer_norm_eps": 1e-5,
                "rope_theta": 10000.0,
            }
            tensor_names = []
            tensor_shapes = {}
            tensor_types = {}
            tensor_data = {}

            tensor_names.append("token_embd.weight")
            tensor_shapes["token_embd.weight"] = (V, H)
            tensor_types["token_embd.weight"] = GGML_TYPE_F32
            tensor_data["token_embd.weight"] = rng.standard_normal((V, H)).astype(np.float32) * 0.02

            for l in range(2):
                p = f"blk.{l}."
                for suf, shape in [
                    ("attn_q.weight", (H, H)),
                    ("attn_k.weight", (H, H)),
                    ("attn_v.weight", (H, H)),
                    ("attn_output.weight", (H, H)),
                    ("attn_norm.weight", (H,)),
                    ("ffn_gate.weight", (I, H)),
                    ("ffn_up.weight", (I, H)),
                    ("ffn_down.weight", (H, I)),
                    ("ffn_norm.weight", (H,)),
                ]:
                    name = p + suf
                    tensor_names.append(name)
                    tensor_shapes[name] = shape
                    tensor_types[name] = GGML_TYPE_F32
                    tensor_data[name] = rng.standard_normal(shape).astype(np.float32) * 0.02

            tensor_names.append("output.weight")
            tensor_shapes["output.weight"] = (V, H)
            tensor_types["output.weight"] = GGML_TYPE_F32
            tensor_data["output.weight"] = rng.standard_normal((V, H)).astype(np.float32) * 0.02

            tensor_names.append("output_norm.weight")
            tensor_shapes["output_norm.weight"] = (H,)
            tensor_types["output_norm.weight"] = GGML_TYPE_F32
            tensor_data["output_norm.weight"] = np.ones(H, dtype=np.float32)

            create_synthetic_gguf(
                gguf_path, config=config, tensor_names=tensor_names,
                tensor_shapes=tensor_shapes, tensor_types=tensor_types,
                tensor_data=tensor_data,
            )

            engine = NativeInferenceEngine(node_id="out-test")
            shard = engine.load_gguf_model("out-model", gguf_path)
            assert shard.loaded

            output_proj = engine.output_proj.get("out-model")
            assert output_proj is not None
            assert output_proj.shape == (V, H)

            ln_f = engine.layer_norm_f.get("out-model")
            assert ln_f is not None
            assert ln_f[0].shape == (H,)

            tokens = [1, 2, 3]
            input_ids = np.array(tokens, dtype=np.int64).reshape(1, -1)
            hidden = engine.embed_tokens["out-model"][input_ids]
            for i in range(2):
                hidden = engine.forward_layer(hidden, "out-model", i)

            normed = _layer_norm(hidden[:, -1, :], ln_f[0], ln_f[1], 1e-5)
            logits = normed @ output_proj.T
            assert logits.shape == (1, V)

    def test_load_gguf_model_f16_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "f16_model.gguf")
            rng = np.random.default_rng(42)
            V, H, I = 60, 20, 40
            config = {
                "hidden_size": H,
                "num_layers": 1,
                "vocab_size": V,
                "intermediate_size": I,
                "num_heads": 4,
                "max_position_embeddings": 32,
                "layer_norm_eps": 1e-5,
                "rope_theta": 10000.0,
            }
            tensor_names = []
            tensor_shapes = {}
            tensor_types = {}
            tensor_data = {}

            tensor_names.append("token_embd.weight")
            tensor_shapes["token_embd.weight"] = (V, H)
            tensor_types["token_embd.weight"] = GGML_TYPE_F16
            tensor_data["token_embd.weight"] = rng.standard_normal((V, H)).astype(np.float16)

            p = "blk.0."
            for suf, shape in [
                ("attn_q.weight", (H, H)),
                ("attn_k.weight", (H, H)),
                ("attn_v.weight", (H, H)),
                ("attn_output.weight", (H, H)),
                ("attn_norm.weight", (H,)),
                ("ffn_gate.weight", (I, H)),
                ("ffn_up.weight", (I, H)),
                ("ffn_down.weight", (H, I)),
                ("ffn_norm.weight", (H,)),
            ]:
                name = p + suf
                tensor_names.append(name)
                tensor_shapes[name] = shape
                tensor_types[name] = GGML_TYPE_F16
                tensor_data[name] = rng.standard_normal(shape).astype(np.float16)

            tensor_names.append("output_norm.weight")
            tensor_shapes["output_norm.weight"] = (H,)
            tensor_types["output_norm.weight"] = GGML_TYPE_F16
            tensor_data["output_norm.weight"] = np.ones(H, dtype=np.float16)

            create_synthetic_gguf(
                gguf_path, config=config, tensor_names=tensor_names,
                tensor_shapes=tensor_shapes, tensor_types=tensor_types,
                tensor_data=tensor_data,
            )

            engine = NativeInferenceEngine(node_id="f16-test")
            shard = engine.load_gguf_model("f16-model", gguf_path)
            assert shard.loaded
            assert shard.num_layers == 1

            hidden = np.random.randn(1, 4, H).astype(np.float32)
            out = engine.forward_layer(hidden, "f16-model", 0)
            assert out.shape == hidden.shape
            assert out.dtype == np.float32

    def test_load_gguf_model_i8_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "i8_model.gguf")
            rng = np.random.default_rng(42)
            V, H, I = 50, 16, 32
            config = {
                "hidden_size": H,
                "num_layers": 1,
                "vocab_size": V,
                "intermediate_size": I,
                "num_heads": 2,
                "max_position_embeddings": 32,
                "layer_norm_eps": 1e-5,
                "rope_theta": 10000.0,
            }
            tensor_names = []
            tensor_shapes = {}
            tensor_types = {}
            tensor_data = {}

            tensor_names.append("token_embd.weight")
            tensor_shapes["token_embd.weight"] = (V, H)
            tensor_types["token_embd.weight"] = GGML_TYPE_I8
            tensor_data["token_embd.weight"] = rng.integers(-8, 8, (V, H)).astype(np.int8)

            p = "blk.0."
            for suf, shape in [
                ("attn_q.weight", (H, H)),
                ("attn_k.weight", (H, H)),
                ("attn_v.weight", (H, H)),
                ("attn_output.weight", (H, H)),
                ("attn_norm.weight", (H,)),
                ("ffn_gate.weight", (I, H)),
                ("ffn_up.weight", (I, H)),
                ("ffn_down.weight", (H, I)),
                ("ffn_norm.weight", (H,)),
            ]:
                name = p + suf
                tensor_names.append(name)
                tensor_shapes[name] = shape
                tensor_types[name] = GGML_TYPE_I8
                tensor_data[name] = rng.integers(-8, 8, shape).astype(np.int8)

            tensor_names.append("output_norm.weight")
            tensor_shapes["output_norm.weight"] = (H,)
            tensor_types["output_norm.weight"] = GGML_TYPE_I8
            tensor_data["output_norm.weight"] = np.ones(H, dtype=np.int8)

            create_synthetic_gguf(
                gguf_path, config=config, tensor_names=tensor_names,
                tensor_shapes=tensor_shapes, tensor_types=tensor_types,
                tensor_data=tensor_data,
            )

            engine = NativeInferenceEngine(node_id="i8-test")
            shard = engine.load_gguf_model("i8-model", gguf_path)
            assert shard.loaded
            assert shard.num_layers == 1

            embed = engine.embed_tokens.get("i8-model")
            assert embed is not None
            assert embed.dtype == np.float32

    def test_unload_gguf_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "unload_model.gguf")
            create_synthetic_gguf(gguf_path)

            engine = NativeInferenceEngine(node_id="unload-test")
            shard = engine.load_gguf_model("unload-model", gguf_path)
            assert shard.loaded
            assert "unload-model" in engine._loaded_models

            engine.unload_model("unload-model")
            assert "unload-model" not in engine._loaded_models
            assert "unload-model" not in engine.configs
            assert "unload-model" not in engine.embed_tokens
            assert "unload-model" not in engine.output_proj
            assert "unload-model" not in engine.layer_norm_f

    def test_gguf_load_from_engine_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "integration.gguf")
            create_synthetic_gguf(gguf_path)

            from netai.inference.engine import InferenceEngine
            ie = InferenceEngine(node_id="gguf-int-test")
            result = ie.load_local_model("gguf-int-model", gguf_path)
            assert result["loaded_layers"] == 2
            assert result["memory_mb"] > 0

            async def _do_infer():
                return await ie.native_infer(
                    model_id="gguf-int-model",
                    prompt_tokens=[1, 2, 3],
                    max_tokens=5,
                    temperature=0.7,
                )

            import asyncio
            infer_result = asyncio.run(_do_infer())
            assert "tokens" in infer_result
            assert infer_result["num_generated"] == 5

    def test_gguf_model_metadata_vocab_from_array(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "vocab_model.gguf")
            config = {
                "hidden_size": 32,
                "num_layers": 1,
                "vocab_size": 50,
                "intermediate_size": 64,
                "num_heads": 4,
                "max_position_embeddings": 64,
                "layer_norm_eps": 1e-5,
                "rope_theta": 10000.0,
            }
            rng = np.random.default_rng(42)
            tensor_names = []
            tensor_shapes = {}
            tensor_types = {}
            tensor_data = {}

            tensor_names.append("token_embd.weight")
            tensor_shapes["token_embd.weight"] = (50, 32)
            tensor_types["token_embd.weight"] = GGML_TYPE_F32
            tensor_data["token_embd.weight"] = rng.standard_normal((50, 32)).astype(np.float32) * 0.02

            p = "blk.0."
            for suf, shape in [
                ("attn_q.weight", (32, 32)),
                ("attn_k.weight", (32, 32)),
                ("attn_v.weight", (32, 32)),
                ("attn_output.weight", (32, 32)),
                ("attn_norm.weight", (32,)),
                ("ffn_gate.weight", (64, 32)),
                ("ffn_up.weight", (64, 32)),
                ("ffn_down.weight", (32, 64)),
                ("ffn_norm.weight", (32,)),
            ]:
                name = p + suf
                tensor_names.append(name)
                tensor_shapes[name] = shape
                tensor_types[name] = GGML_TYPE_F32
                tensor_data[name] = rng.standard_normal(shape).astype(np.float32) * 0.02

            tensor_names.append("output_norm.weight")
            tensor_shapes["output_norm.weight"] = (32,)
            tensor_types["output_norm.weight"] = GGML_TYPE_F32
            tensor_data["output_norm.weight"] = np.ones(32, dtype=np.float32)

            extra = [
                ("tokenizer.ggml.tokens", [f"tok_{i}" for i in range(50)]),
            ]

            create_synthetic_gguf(
                gguf_path, config=config, tensor_names=tensor_names,
                tensor_shapes=tensor_shapes, tensor_types=tensor_types,
                tensor_data=tensor_data, extra_metadata=extra,
            )

            engine = NativeInferenceEngine(node_id="vocab-test")
            shard = engine.load_gguf_model("vocab-model", gguf_path)
            assert shard.loaded
            assert engine.configs["vocab-model"].vocab_size == 50

    def test_gguf_embed_and_output_share(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "share_model.gguf")
            rng = np.random.default_rng(42)
            V, H, I = 60, 24, 48
            config = {
                "hidden_size": H,
                "num_layers": 1,
                "vocab_size": V,
                "intermediate_size": I,
                "num_heads": 4,
                "max_position_embeddings": 32,
                "layer_norm_eps": 1e-5,
                "rope_theta": 10000.0,
            }
            tensor_names = []
            tensor_shapes = {}
            tensor_types = {}
            tensor_data = {}

            emb = rng.standard_normal((V, H)).astype(np.float32) * 0.02
            tensor_names.append("token_embd.weight")
            tensor_shapes["token_embd.weight"] = (V, H)
            tensor_types["token_embd.weight"] = GGML_TYPE_F32
            tensor_data["token_embd.weight"] = emb

            p = "blk.0."
            for suf, shape in [
                ("attn_q.weight", (H, H)),
                ("attn_k.weight", (H, H)),
                ("attn_v.weight", (H, H)),
                ("attn_output.weight", (H, H)),
                ("attn_norm.weight", (H,)),
                ("ffn_gate.weight", (I, H)),
                ("ffn_up.weight", (I, H)),
                ("ffn_down.weight", (H, I)),
                ("ffn_norm.weight", (H,)),
            ]:
                name = p + suf
                tensor_names.append(name)
                tensor_shapes[name] = shape
                tensor_types[name] = GGML_TYPE_F32
                tensor_data[name] = rng.standard_normal(shape).astype(np.float32) * 0.02

            tensor_names.append("output.weight")
            tensor_shapes["output.weight"] = (V, H)
            tensor_types["output.weight"] = GGML_TYPE_F32
            tensor_data["output.weight"] = emb.copy()

            tensor_names.append("output_norm.weight")
            tensor_shapes["output_norm.weight"] = (H,)
            tensor_types["output_norm.weight"] = GGML_TYPE_F32
            tensor_data["output_norm.weight"] = np.ones(H, dtype=np.float32)

            create_synthetic_gguf(
                gguf_path, config=config, tensor_names=tensor_names,
                tensor_shapes=tensor_shapes, tensor_types=tensor_types,
                tensor_data=tensor_data,
            )

            engine = NativeInferenceEngine(node_id="share-test")
            shard = engine.load_gguf_model("share-model", gguf_path)
            assert shard.loaded

            embed = engine.embed_tokens["share-model"]
            output = engine.output_proj["share-model"]
            assert embed.shape == (V, H)
            assert output.shape == (V, H)

    def test_gguf_status_reflects_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "status_model.gguf")
            create_synthetic_gguf(gguf_path)

            engine = NativeInferenceEngine(node_id="status-test")
            engine.load_gguf_model("status-model", gguf_path)

            status = engine.get_status()
            assert "status-model" in status["loaded_models"]
            assert status["num_layers_loaded"] == 2
            assert status["weights_memory_mb"] > 0
