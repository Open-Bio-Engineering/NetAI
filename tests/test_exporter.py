import os
import tempfile

from netai.inference.native_engine import NativeInferenceEngine, TransformerConfig
from netai.sharing.exporter import ModelExporter


class TestModelExporter:
    def test_init(self):
        exporter = ModelExporter()
        assert "npz" in exporter.SUPPORTED_FORMATS

    def test_export_formats_list(self):
        exporter = ModelExporter()
        formats = exporter.get_export_formats()
        assert len(formats) >= 3

    def test_npz_export_roundtrip(self):
        engine = NativeInferenceEngine(node_id="export-test")
        config = TransformerConfig(hidden_size=32, num_layers=2, num_heads=2,
                                   vocab_size=100, intermediate_size=64, model_type="gpt2")
        engine.configs["export-model"] = config
        engine._loaded_models.add("export-model")
        import numpy as np
        engine.embed_tokens["export-model"] = np.random.randn(100, 32).astype(np.float32) * 0.02
        engine.output_proj["export-model"] = np.random.randn(100, 32).astype(np.float32) * 0.02
        for i in range(2):
            engine.layers[f"export-model/layer_{i}"] = {
                f"attn.c_attn.weight": np.random.randn(32, 96).astype(np.float32) * 0.02,
                f"attn.c_proj.weight": np.random.randn(32, 32).astype(np.float32) * 0.02,
                f"mlp.c_fc.weight": np.random.randn(32, 64).astype(np.float32) * 0.02,
                f"mlp.c_proj.weight": np.random.randn(64, 32).astype(np.float32) * 0.02,
            }

        exporter = ModelExporter(engine=engine)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "model.npz")
            result = exporter.export_to_npz("export-model", out)
            assert result["format"] == "npz"
            assert result["path"] == out
            assert result["tensors"] == 2 * 4 + 2
            assert os.path.exists(out)

    def test_config_export(self):
        engine = NativeInferenceEngine(node_id="cfg-test")
        config = TransformerConfig(hidden_size=64, num_layers=4, num_heads=4,
                                   vocab_size=200, model_type="gpt2")
        engine.configs["cfg-model"] = config
        engine._loaded_models.add("cfg-model")

        exporter = ModelExporter(engine=engine)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "config.json")
            result = exporter.export_config("cfg-model", out)
            assert result["format"] == "config"
            assert os.path.exists(out)

    def test_gguf_export(self):
        engine = NativeInferenceEngine(node_id="gguf-test")
        config = TransformerConfig(hidden_size=32, num_layers=2, num_heads=2,
                                   vocab_size=100, intermediate_size=64, model_type="gpt2")
        engine.configs["gguf-model"] = config
        engine._loaded_models.add("gguf-model")

        exporter = ModelExporter(engine=engine)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "model.gguf")
            result = exporter.export_to_gguf("gguf-model", out, quant_type="f32")
            assert result["format"] == "gguf"
            assert result["model_id"] == "gguf-model"
            assert os.path.exists(out)

    def test_edge_export(self):
        engine = NativeInferenceEngine(node_id="edge-test")
        config = TransformerConfig(hidden_size=32, num_layers=1, num_heads=2,
                                   vocab_size=50, intermediate_size=64, model_type="gpt2")
        engine.configs["edge-model"] = config
        engine._loaded_models.add("edge-model")

        exporter = ModelExporter(engine=engine)
        with tempfile.TemporaryDirectory() as tmpdir:
            edge_dir = os.path.join(tmpdir, "edge")
            result = exporter.export_for_edge("edge-model", edge_dir, quant="q4_0")
            assert "gguf" in str(result.get("format", ""))
            assert os.path.isdir(edge_dir)

    def test_export_not_loaded(self):
        engine = NativeInferenceEngine(node_id="no-load")
        exporter = ModelExporter(engine=engine)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "missing.npz")
            result = exporter.export_to_npz("nonexistent", out)
            assert "error" in result

    def test_export_no_config(self):
        engine = NativeInferenceEngine(node_id="no-cfg")
        engine._loaded_models.add("no-cfg-model")
        exporter = ModelExporter(engine=engine)
        result = exporter.export_config("no-cfg-model", "/tmp/test.json")
        assert "error" in result
