"""Multi-architecture support: OPT, Phi, Qwen2, Gemma, LLaMA, GPT-2.

Defines architecture traits, weight name mappings, and activation functions
for each supported model family. Extends NativeInferenceEngine."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import logging

logger = logging.getLogger(__name__)


class NormPosition(str, Enum):
    PRE = "pre"
    POST = "post"


class ActivationFn(str, Enum):
    GELU = "gelu"
    SILU = "silu"
    RELU = "relu"
    GEGLU = "geglu"


class PositionEncoding(str, Enum):
    LEARNED = "learned"
    ROPE = "rope"
    ALIBI = "alibi"


class AttentionQKV(str, Enum):
    COMBINED = "combined"
    SEPARATE = "separate"


@dataclass
class ArchitectureTraits:
    norm_position: NormPosition = NormPosition.POST
    activation: ActivationFn = ActivationFn.GELU
    position_encoding: PositionEncoding = PositionEncoding.LEARNED
    attention_qkv: AttentionQKV = AttentionQKV.COMBINED
    has_bias: bool = True
    ffn_type: str = "standard"
    tie_word_embeddings: bool = True
    partial_rope: bool = False


ARCHITECTURE_CONFIGS: dict[str, ArchitectureTraits] = {
    "gpt2": ArchitectureTraits(
        norm_position=NormPosition.POST,
        activation=ActivationFn.GELU,
        position_encoding=PositionEncoding.LEARNED,
        attention_qkv=AttentionQKV.COMBINED,
        has_bias=True, ffn_type="standard", tie_word_embeddings=True,
    ),
    "gpt_neo": ArchitectureTraits(
        norm_position=NormPosition.POST,
        activation=ActivationFn.GELU,
        position_encoding=PositionEncoding.LEARNED,
        attention_qkv=AttentionQKV.COMBINED,
        has_bias=True, ffn_type="standard", tie_word_embeddings=True,
    ),
    "llama": ArchitectureTraits(
        norm_position=NormPosition.PRE,
        activation=ActivationFn.SILU,
        position_encoding=PositionEncoding.ROPE,
        attention_qkv=AttentionQKV.SEPARATE,
        has_bias=False, ffn_type="gated", tie_word_embeddings=False,
    ),
    "mistral": ArchitectureTraits(
        norm_position=NormPosition.PRE,
        activation=ActivationFn.SILU,
        position_encoding=PositionEncoding.ROPE,
        attention_qkv=AttentionQKV.SEPARATE,
        has_bias=False, ffn_type="gated", tie_word_embeddings=False,
    ),
    "opt": ArchitectureTraits(
        norm_position=NormPosition.PRE,
        activation=ActivationFn.RELU,
        position_encoding=PositionEncoding.LEARNED,
        attention_qkv=AttentionQKV.SEPARATE,
        has_bias=False, ffn_type="standard", tie_word_embeddings=True,
    ),
    "phi3": ArchitectureTraits(
        norm_position=NormPosition.PRE,
        activation=ActivationFn.GELU,
        position_encoding=PositionEncoding.ROPE,
        attention_qkv=AttentionQKV.COMBINED,
        has_bias=True, ffn_type="standard", tie_word_embeddings=False,
        partial_rope=True,
    ),
    "phi": ArchitectureTraits(
        norm_position=NormPosition.PRE,
        activation=ActivationFn.GELU,
        position_encoding=PositionEncoding.ROPE,
        attention_qkv=AttentionQKV.COMBINED,
        has_bias=True, ffn_type="standard", tie_word_embeddings=False,
        partial_rope=True,
    ),
    "qwen2": ArchitectureTraits(
        norm_position=NormPosition.PRE,
        activation=ActivationFn.SILU,
        position_encoding=PositionEncoding.ROPE,
        attention_qkv=AttentionQKV.SEPARATE,
        has_bias=True, ffn_type="gated", tie_word_embeddings=False,
    ),
    "qwen2_moe": ArchitectureTraits(
        norm_position=NormPosition.PRE,
        activation=ActivationFn.SILU,
        position_encoding=PositionEncoding.ROPE,
        attention_qkv=AttentionQKV.SEPARATE,
        has_bias=True, ffn_type="gated", tie_word_embeddings=False,
    ),
    "gemma": ArchitectureTraits(
        norm_position=NormPosition.PRE,
        activation=ActivationFn.GEGLU,
        position_encoding=PositionEncoding.ROPE,
        attention_qkv=AttentionQKV.SEPARATE,
        has_bias=False, ffn_type="gated", tie_word_embeddings=False,
    ),
    "gemma2": ArchitectureTraits(
        norm_position=NormPosition.PRE,
        activation=ActivationFn.GELU,
        position_encoding=PositionEncoding.ROPE,
        attention_qkv=AttentionQKV.SEPARATE,
        has_bias=False, ffn_type="gated", tie_word_embeddings=False,
    ),
    "stablelm": ArchitectureTraits(
        norm_position=NormPosition.PRE,
        activation=ActivationFn.SILU,
        position_encoding=PositionEncoding.ROPE,
        attention_qkv=AttentionQKV.SEPARATE,
        has_bias=False, ffn_type="gated", tie_word_embeddings=False,
    ),
}


LAYER_PREFIX_PATTERNS: dict[str, list[str]] = {
    "gpt2": ["transformer.h.", "h.", "layers.", "block."],
    "gpt_neo": ["transformer.h.", "h.", "layers.", "block."],
    "opt": ["model.decoder.layers.", "decoder.layers.", "model.layers.", "layers.", "block."],
    "phi3": ["model.layers.", "transformer.h.", "h.", "layers.", "block."],
    "phi": ["model.layers.", "transformer.h.", "h.", "layers.", "block."],
    "llama": ["model.layers.", "transformer.h.", "h.", "layers.", "block."],
    "mistral": ["model.layers.", "transformer.h.", "h.", "layers.", "block."],
    "qwen2": ["model.layers.", "transformer.h.", "h.", "layers.", "block."],
    "qwen2_moe": ["model.layers.", "transformer.h.", "h.", "layers.", "block."],
    "gemma": ["model.layers.", "transformer.h.", "h.", "layers.", "block."],
    "gemma2": ["model.layers.", "transformer.h.", "h.", "layers.", "block."],
    "stablelm": ["model.layers.", "transformer.h.", "h.", "layers.", "block."],
}


WEIGHT_KEY_MAPS: dict[str, dict[str, list[str]]] = {
    "gpt2": {
        "embed": ["wte.weight", "transformer.wte.weight"],
        "output_proj": ["wte.weight", "lm_head.weight"],
        "ln_f": ["ln_f.weight", "transformer.ln_f.weight", "model.norm.weight"],
        "attn_qkv": ["attn.c_attn.weight"],
        "attn_out": ["attn.c_proj.weight"],
        "ffn_up": ["mlp.c_fc.weight"],
        "ffn_down": ["mlp.c_proj.weight"],
        "ln1_w": ["ln_1.weight", "input_layernorm.weight"],
        "ln2_w": ["ln_2.weight", "post_attention_layernorm.weight"],
    },
    "opt": {
        "embed": ["model.decoder.embed_tokens.weight", "decoder.embed_tokens.weight", "wte.weight"],
        "output_proj": ["lm_head.weight", "model.decoder.embed_tokens.weight", "wte.weight"],
        "ln_f": ["model.decoder.final_layer_norm.weight", "model.norm.weight", "ln_f.weight"],
        "attn_q": ["self_attn.q_proj.weight", "attn.q_proj.weight"],
        "attn_k": ["self_attn.k_proj.weight", "attn.k_proj.weight"],
        "attn_v": ["self_attn.v_proj.weight", "attn.v_proj.weight"],
        "attn_out": ["self_attn.out_proj.weight", "attn.c_proj.weight"],
        "ffn_up": ["fc1.weight", "mlp.fc1.weight", "mlp.up_proj.weight"],
        "ffn_down": ["fc2.weight", "mlp.fc2.weight", "mlp.down_proj.weight"],
        "ln1_w": ["self_attn_layer_norm.weight", "input_layernorm.weight", "ln_1.weight"],
        "ln2_w": ["final_layer_norm.weight", "post_attention_layernorm.weight", "ln_2.weight"],
    },
    "phi3": {
        "embed": ["model.embed_tokens.weight", "wte.weight"],
        "output_proj": ["lm_head.weight"],
        "ln_f": ["model.norm.weight", "ln_f.weight"],
        "attn_qkv": ["self_attn.qkv_proj.weight", "attn.qkv_proj.weight", "attn.c_attn.weight"],
        "attn_out": ["self_attn.o_proj.weight", "attn.c_proj.weight"],
        "ffn_up": ["mlp.fc1.weight", "mlp.c_fc.weight", "mlp.up_proj.weight"],
        "ffn_down": ["mlp.fc2.weight", "mlp.c_proj.weight", "mlp.down_proj.weight"],
        "ln1_w": ["input_layernorm.weight", "ln_1.weight"],
        "ln2_w": ["post_attention_layernorm.weight", "ln_2.weight"],
    },
    "llama": {
        "embed": ["model.embed_tokens.weight", "wte.weight"],
        "output_proj": ["lm_head.weight"],
        "ln_f": ["model.norm.weight", "ln_f.weight"],
        "attn_q": ["self_attn.q_proj.weight", "attn.q_proj.weight"],
        "attn_k": ["self_attn.k_proj.weight", "attn.k_proj.weight"],
        "attn_v": ["self_attn.v_proj.weight", "attn.v_proj.weight"],
        "attn_out": ["self_attn.o_proj.weight", "attn.c_proj.weight"],
        "ffn_gate": ["mlp.gate_proj.weight"],
        "ffn_up": ["mlp.up_proj.weight", "mlp.c_fc.weight"],
        "ffn_down": ["mlp.down_proj.weight", "mlp.c_proj.weight"],
        "ln1_w": ["input_layernorm.weight", "ln_1.weight"],
        "ln2_w": ["post_attention_layernorm.weight", "ln_2.weight"],
    },
    "qwen2": {
        "embed": ["model.embed_tokens.weight", "wte.weight"],
        "output_proj": ["lm_head.weight"],
        "ln_f": ["model.norm.weight", "ln_f.weight"],
        "attn_q": ["self_attn.q_proj.weight", "attn.q_proj.weight"],
        "attn_k": ["self_attn.k_proj.weight", "attn.k_proj.weight"],
        "attn_v": ["self_attn.v_proj.weight", "attn.v_proj.weight"],
        "attn_out": ["self_attn.o_proj.weight", "attn.c_proj.weight"],
        "ffn_gate": ["mlp.gate_proj.weight"],
        "ffn_up": ["mlp.up_proj.weight", "mlp.c_fc.weight"],
        "ffn_down": ["mlp.down_proj.weight", "mlp.c_proj.weight"],
        "ln1_w": ["input_layernorm.weight", "ln_1.weight"],
        "ln2_w": ["post_attention_layernorm.weight", "ln_2.weight"],
    },
    "gemma": {
        "embed": ["model.embed_tokens.weight", "wte.weight"],
        "output_proj": ["lm_head.weight"],
        "ln_f": ["model.norm.weight", "ln_f.weight"],
        "attn_q": ["self_attn.q_proj.weight", "attn.q_proj.weight"],
        "attn_k": ["self_attn.k_proj.weight", "attn.k_proj.weight"],
        "attn_v": ["self_attn.v_proj.weight", "attn.v_proj.weight"],
        "attn_out": ["self_attn.o_proj.weight", "attn.c_proj.weight"],
        "ffn_gate": ["mlp.gate_proj.weight"],
        "ffn_up": ["mlp.up_proj.weight", "mlp.c_fc.weight"],
        "ffn_down": ["mlp.down_proj.weight", "mlp.c_proj.weight"],
        "ln1_w": ["input_layernorm.weight", "ln_1.weight"],
        "ln2_w": ["post_attention_layernorm.weight", "ln_2.weight"],
    },
}


def get_architecture_traits(model_type: str) -> ArchitectureTraits:
    if model_type in ARCHITECTURE_CONFIGS:
        return ARCHITECTURE_CONFIGS[model_type]
    if model_type == "gpt2" or model_type.startswith("gpt"):
        return ArchitectureTraits()
    if "phi" in model_type.lower():
        return ARCHITECTURE_CONFIGS["phi3"]
    if "qwen" in model_type.lower():
        return ARCHITECTURE_CONFIGS["qwen2"]
    if "gemma" in model_type.lower():
        return ARCHITECTURE_CONFIGS["gemma"]
    if "llama" in model_type.lower():
        return ARCHITECTURE_CONFIGS["llama"]
    if "mistral" in model_type.lower():
        return ARCHITECTURE_CONFIGS["mistral"]
    if "opt" in model_type.lower():
        return ARCHITECTURE_CONFIGS["opt"]
    return ArchitectureTraits()


def get_layer_prefixes(model_type: str) -> list[str]:
    return LAYER_PREFIX_PATTERNS.get(model_type, ["transformer.h.", "h.", "model.layers.", "layers.", "block."])


def get_weight_keys(model_type: str) -> dict[str, list[str]]:
    if model_type in WEIGHT_KEY_MAPS:
        return WEIGHT_KEY_MAPS[model_type]
    if "opt" in model_type.lower():
        return WEIGHT_KEY_MAPS["opt"]
    if "phi" in model_type.lower():
        return WEIGHT_KEY_MAPS["phi3"]
    if "qwen" in model_type.lower():
        return WEIGHT_KEY_MAPS["qwen2"]
    if "gemma" in model_type.lower():
        return WEIGHT_KEY_MAPS["gemma"]
    if "llama" in model_type.lower() or "mistral" in model_type.lower():
        return WEIGHT_KEY_MAPS["llama"]
    return WEIGHT_KEY_MAPS["gpt2"]


def _apply_relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def get_activation_fn(traits: ArchitectureTraits):
    if traits.activation == ActivationFn.GELU or traits.activation == ActivationFn.GEGLU:
        from netai.inference.native_engine import _gelu
        return _gelu
    elif traits.activation == ActivationFn.SILU:
        from netai.inference.native_engine import _silu
        return _silu
    elif traits.activation == ActivationFn.RELU:
        return _apply_relu
    return _apply_relu
