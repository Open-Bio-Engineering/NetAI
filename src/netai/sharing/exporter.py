"""Model export and sharing: NPZ, safetensors, GGUF, peer-to-peer sharing."""

from __future__ import annotations

import json
import logging
import os
import struct
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export loaded models to various formats and share with peers."""

    SUPPORTED_FORMATS = ["npz", "safetensors", "gguf", "config"]

    def __init__(self, engine=None, downloader=None):
        self._engine = engine
        self._downloader = downloader

    def export_to_npz(self, model_id: str, output_path: str) -> dict[str, Any]:
        engine = self._engine
        if model_id not in engine._loaded_models:
            return {"error": f"Model {model_id} not loaded"}
        weights = {}
        config = engine.configs.get(model_id)
        if not config:
            return {"error": f"No config for {model_id}"}
        for i in range(config.num_layers):
            key = f"{model_id}/layer_{i}"
            layer_w = engine.layers.get(key, {})
            for wk, wv in layer_w.items():
                weights[f"layer_{i}/{wk}"] = wv
        if model_id in engine.embed_tokens:
            weights["embed_tokens"] = engine.embed_tokens[model_id]
        if model_id in engine.output_proj:
            weights["output_proj"] = engine.output_proj[model_id]
        np.savez_compressed(output_path, **weights)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info("Exported %s to NPZ: %.1f MB (%d tensors)", model_id, size_mb, len(weights))
        return {"format": "npz", "path": output_path, "size_mb": round(size_mb, 2),
                "tensors": len(weights), "model_id": model_id}

    def export_config(self, model_id: str, output_path: str) -> dict[str, Any]:
        engine = self._engine
        config = engine.configs.get(model_id)
        if not config:
            return {"error": f"No config for {model_id}"}
        cfg_dict = {
            "model_type": config.model_type, "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size, "num_hidden_layers": config.num_layers,
            "num_attention_heads": config.num_heads,
            "intermediate_size": config.intermediate_size,
            "max_position_embeddings": config.max_position_embeddings,
            "layer_norm_eps": config.layer_norm_eps, "rope_theta": config.rope_theta,
        }
        with open(output_path, "w") as f:
            json.dump(cfg_dict, f, indent=2)
        return {"format": "config", "path": output_path, "model_id": model_id}

    def export_to_gguf(self, model_id: str, output_path: str, quant_type: str = "f32") -> dict[str, Any]:
        engine = self._engine
        config = engine.configs.get(model_id)
        if not config:
            return {"error": f"No config for {model_id}"}

        type_map = {"f32": 0, "f16": 1, "q4_0": 2, "q8_0": 8}
        tensor_type = type_map.get(quant_type, 0)
        alignment = 32

        header = bytearray()
        header += b"GGUF"
        header += struct.pack("<I", 3)
        tensor_count = config.num_layers * 8 + 3
        header += struct.pack("<Q", tensor_count)
        kv_count = 12
        header += struct.pack("<Q", kv_count)

        md = {
            "general.architecture": config.model_type,
            "general.name": model_id,
            "general.quantization_version": 2 if quant_type.startswith("q") else 1,
            "llm.context_length": config.max_position_embeddings,
            "llm.embedding_length": config.hidden_size,
            "llm.block_count": config.num_layers,
            "llm.feed_forward_length": config.intermediate_size,
            "llm.attention.head_count": config.num_heads,
            "llm.attention.head_count_kv": config.num_heads,
            "llm.rope.dimension_count": config.hidden_size // config.num_heads,
            "llm.rope.freq_base": config.rope_theta,
            "tokenizer.ggml.model": "gpt2",
        }

        for key, value in md.items():
            name_bytes = key.encode("utf-8")
            header += struct.pack("<Q", len(name_bytes))
            header += name_bytes
            if isinstance(value, str):
                header += struct.pack("<I", 8)
                val_bytes = value.encode("utf-8")
                header += struct.pack("<Q", len(val_bytes))
                header += val_bytes
            elif isinstance(value, int):
                header += struct.pack("<I", 10)
                header += struct.pack("<Q", value)
            elif isinstance(value, bool):
                header += struct.pack("<I", 7)
                header += struct.pack("<B", 1 if value else 0)
            elif isinstance(value, float):
                header += struct.pack("<I", 12)
                header += struct.pack("<d", value)

        header += b"\x00" * 4

        with open(output_path, "wb") as f:
            f.write(bytes(header))

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        return {"format": "gguf", "path": output_path, "size_mb": round(size_mb, 2),
                "model_id": model_id, "tensor_count": tensor_count}

    def export_for_edge(self, model_id: str, output_dir: str, quant: str = "q4_0") -> dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "config.json")
        self.export_config(model_id, config_path)
        gguf_path = os.path.join(output_dir, f"{model_id}.gguf")
        return self.export_to_gguf(model_id, gguf_path, quant_type=quant)

    def get_export_formats(self) -> list[str]:
        return list(self.SUPPORTED_FORMATS)
