from netai.inference.architectures import (
    get_architecture_traits, get_layer_prefixes, get_weight_keys,
    get_activation_fn, ArchitectureTraits, ARCHITECTURE_CONFIGS,
    NormPosition, ActivationFn, PositionEncoding, LAYER_PREFIX_PATTERNS,
    WEIGHT_KEY_MAPS,
)
import numpy as np


class TestArchitectureTraits:
    def test_gpt2_traits(self):
        t = get_architecture_traits("gpt2")
        assert t.norm_position == NormPosition.POST
        assert t.activation == ActivationFn.GELU
        assert t.attention_qkv == "combined"
        assert t.tie_word_embeddings

    def test_llama_traits(self):
        t = get_architecture_traits("llama")
        assert t.norm_position == NormPosition.PRE
        assert t.activation == ActivationFn.SILU
        assert t.attention_qkv == "separate"
        assert t.ffn_type == "gated"
        assert not t.has_bias

    def test_opt_traits(self):
        t = get_architecture_traits("opt")
        assert t.norm_position == NormPosition.PRE
        assert t.activation == ActivationFn.RELU
        assert not t.has_bias

    def test_phi3_traits(self):
        t = get_architecture_traits("phi3")
        assert t.norm_position == NormPosition.PRE
        assert t.activation == ActivationFn.GELU
        assert t.attention_qkv == "combined"
        assert t.partial_rope

    def test_qwen2_traits(self):
        t = get_architecture_traits("qwen2")
        assert t.activation == ActivationFn.SILU
        assert t.ffn_type == "gated"
        assert t.has_bias

    def test_gemma_traits(self):
        t = get_architecture_traits("gemma")
        assert t.activation == ActivationFn.GEGLU
        assert t.ffn_type == "gated"

    def test_unknown_fallback(self):
        t = get_architecture_traits("nonexistent-arch")
        assert t.activation == ActivationFn.GELU
        assert t.norm_position == NormPosition.POST

    def test_fuzzy_matching(self):
        assert get_architecture_traits("opt-125m").activation == ActivationFn.RELU
        assert get_architecture_traits("phi-4-mini").attention_qkv == "combined"
        assert get_architecture_traits("qwen2.5-3b").activation == ActivationFn.SILU
        assert get_architecture_traits("gemma-4-e2b").activation == ActivationFn.GEGLU


class TestWeightKeys:
    def test_gpt2_keys(self):
        keys = get_weight_keys("gpt2")
        assert "embed" in keys
        assert len(keys["embed"]) >= 1

    def test_opt_keys(self):
        keys = get_weight_keys("opt")
        assert "attn_q" in keys
        assert "ffn_up" in keys

    def test_phi3_keys(self):
        keys = get_weight_keys("phi3")
        assert "attn_qkv" in keys

    def test_llama_keys(self):
        keys = get_weight_keys("llama")
        assert "ffn_gate" in keys

    def test_layer_prefixes_all(self):
        for arch in ["gpt2", "opt", "phi3", "llama", "qwen2", "gemma"]:
            prefixes = get_layer_prefixes(arch)
            assert len(prefixes) >= 3

    def test_get_activation_fn_relu(self):
        t = ArchitectureTraits(activation=ActivationFn.RELU)
        fn = get_activation_fn(t)
        x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        result = fn(x)
        assert result[0, 0] == 0.0
        assert result[0, 2] == 1.0
