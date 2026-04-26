"""Tests for ModelRegistry, ModelEntry, ModelSizeClass, and AutoLoader."""

import json
import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from netai.inference.autoloader import (
    AutoLoader, ModelEntry, ModelRegistry, ModelSizeClass,
    CATALOG_URL, LOCAL_CATALOG_PATH, CATALOG_CACHE_PATH,
)

CATALOG_DATA = {
    "version": "1.0.0",
    "models": [
        {
            "model_id": "test-mini",
            "name": "Test Mini Model",
            "architecture": "llama3",
            "size_class": "mini",
            "params_m": 3000,
            "hidden_size": 2048,
            "num_layers": 28,
            "num_heads": 16,
            "vocab_size": 128256,
            "intermediate_size": 8192,
            "quantizations": ["q4_k_m", "q5_k_m", "fp16"],
            "vram_required_mb": {"q4_k_m": 1900, "q5_k_m": 2600, "fp16": 6200},
            "context_length": 131072,
            "license": "apache-2.0",
            "huggingface_id": "test/test-mini",
            "description": "A mini test model",
        },
        {
            "model_id": "test-small",
            "name": "Test Small Model",
            "architecture": "qwen2",
            "size_class": "small",
            "params_m": 7000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "vocab_size": 152064,
            "intermediate_size": 14336,
            "quantizations": ["q4_k_m", "q5_k_m"],
            "vram_required_mb": {"q4_k_m": 4100, "q5_k_m": 5300},
            "context_length": 32768,
            "license": "apache-2.0",
            "huggingface_id": "test/test-small",
            "description": "A small test model",
        },
        {
            "model_id": "test-mid",
            "name": "Test Mid Model",
            "architecture": "llama3",
            "size_class": "mid",
            "params_m": 70000,
            "hidden_size": 8192,
            "num_layers": 80,
            "num_heads": 64,
            "vocab_size": 128256,
            "intermediate_size": 28672,
            "quantizations": ["q4_k_m", "q5_k_m"],
            "vram_required_mb": {"q4_k_m": 40000, "q5_k_m": 52000},
            "context_length": 131072,
            "license": "llama3.1",
            "huggingface_id": "test/test-mid",
            "description": "A mid test model",
        },
        {
            "model_id": "test-large",
            "name": "Test Large Model",
            "architecture": "deepseek",
            "size_class": "large",
            "params_m": 685000,
            "hidden_size": 7168,
            "num_layers": 61,
            "num_heads": 128,
            "vocab_size": 129280,
            "intermediate_size": 18432,
            "quantizations": ["q4_k_m"],
            "vram_required_mb": {"q4_k_m": 385000},
            "context_length": 131072,
            "license": "mit",
            "huggingface_id": "test/test-large",
            "description": "A large test model",
        },
        {
            "model_id": "test-no-size",
            "name": "Test Auto Size",
            "architecture": "gemma4",
            "params_m": 50,
            "hidden_size": 1024,
            "num_layers": 12,
            "num_heads": 8,
            "vocab_size": 256000,
            "intermediate_size": 4096,
            "quantizations": ["q4_k_m"],
            "vram_required_mb": {"q4_k_m": 300},
            "context_length": 8192,
            "license": "gemma",
            "huggingface_id": "test/test-auto",
            "description": "Auto-sized model",
        },
    ],
}


class TestModelSizeClass:
    def test_for_params_mini(self):
        assert ModelSizeClass.for_params(50) == ModelSizeClass.MINI
        assert ModelSizeClass.for_params(100) == ModelSizeClass.MINI
        assert ModelSizeClass.for_params(2) == ModelSizeClass.MINI

    def test_for_params_small(self):
        assert ModelSizeClass.for_params(150) == ModelSizeClass.SMALL
        assert ModelSizeClass.for_params(300) == ModelSizeClass.SMALL

    def test_for_params_mid(self):
        assert ModelSizeClass.for_params(400) == ModelSizeClass.MID
        assert ModelSizeClass.for_params(600) == ModelSizeClass.MID

    def test_for_params_large(self):
        assert ModelSizeClass.for_params(701) == ModelSizeClass.LARGE
        assert ModelSizeClass.for_params(5000) == ModelSizeClass.LARGE

    def test_min_vram_mb(self):
        assert ModelSizeClass.min_vram_mb(ModelSizeClass.MINI) == 0
        assert ModelSizeClass.min_vram_mb(ModelSizeClass.SMALL) == 4000
        assert ModelSizeClass.min_vram_mb(ModelSizeClass.MID) == 17000
        assert ModelSizeClass.min_vram_mb(ModelSizeClass.LARGE) == 40000

    def test_string_enum(self):
        assert ModelSizeClass.MINI.value == "mini"
        assert ModelSizeClass.SMALL.value == "small"
        assert ModelSizeClass.MID.value == "mid"
        assert ModelSizeClass.LARGE.value == "large"


class TestModelEntry:
    def test_basic_construction(self):
        data = CATALOG_DATA["models"][0]
        entry = ModelEntry(data)
        assert entry.model_id == "test-mini"
        assert entry.name == "Test Mini Model"
        assert entry.architecture == "llama3"
        assert entry.size_class == ModelSizeClass.MINI
        assert entry.params_m == 3000
        assert entry.hidden_size == 2048
        assert entry.num_layers == 28
        assert entry.num_heads == 16
        assert entry.vocab_size == 128256
        assert entry.intermediate_size == 8192
        assert "q4_k_m" in entry.quantizations
        assert entry.context_length == 131072
        assert entry.license == "apache-2.0"

    def test_auto_size_class(self):
        data = {
            "model_id": "auto-mini",
            "name": "Auto",
            "architecture": "test",
            "params_m": 50,
            "vram_required_mb": {"q4_k_m": 300},
        }
        entry = ModelEntry(data)
        assert entry.size_class == ModelSizeClass.MINI

    def test_explicit_size_class_overrides_params(self):
        data = dict(CATALOG_DATA["models"][0])
        data["size_class"] = "small"
        entry = ModelEntry(data)
        assert entry.size_class == ModelSizeClass.SMALL

    def test_vram_for_quant(self):
        entry = ModelEntry(CATALOG_DATA["models"][0])
        assert entry.vram_for_quant("q4_k_m") == 1900
        assert entry.vram_for_quant("fp16") == 6200
        assert entry.vram_for_quant("q5_k_m") == 2600

    def test_vram_for_quant_fallback(self):
        data = {
            "model_id": "fb",
            "name": "Fallback",
            "architecture": "test",
            "params_m": 100,
            "vram_required_mb": {"q4_k_m": 500},
        }
        entry = ModelEntry(data)
        assert entry.vram_for_quant("fp16") == 500

    def test_vram_for_quant_empty(self):
        data = {
            "model_id": "empty",
            "name": "Empty",
            "architecture": "test",
            "params_m": 100,
            "vram_required_mb": {},
        }
        entry = ModelEntry(data)
        assert entry.vram_for_quant("q4_k_m") == 0

    def test_can_fit_true(self):
        entry = ModelEntry(CATALOG_DATA["models"][0])
        assert entry.can_fit(5000, "q4_k_m") is True

    def test_can_fit_false_too_much_vram(self):
        entry = ModelEntry(CATALOG_DATA["models"][0])
        assert entry.can_fit(1000, "q4_k_m") is False

    def test_can_fit_with_85_percent_headroom(self):
        entry = ModelEntry(CATALOG_DATA["models"][0])
        vram_for_q4 = 1900
        needed = vram_for_q4 / 0.85
        assert entry.can_fit(needed, "q4_k_m") is True
        assert entry.can_fit(vram_for_q4, "q4_k_m") is False

    def test_to_config_dict(self):
        entry = ModelEntry(CATALOG_DATA["models"][0])
        config = entry.to_config_dict("q4_k_m")
        assert config["model_id"] == "test-mini"
        assert config["model_name"] == "Test Mini Model"
        assert config["num_shards"] == 1
        assert config["quantization"] == "q4_k_m"
        assert config["hidden_size"] == 2048
        assert config["num_layers"] == 28
        assert config["params_m"] == 3000
        assert config["context_length"] == 131072

    def test_default_quantizations(self):
        data = {
            "model_id": "dq",
            "name": "Default Quant",
            "architecture": "test",
            "params_m": 100,
            "vram_required_mb": {"q4_k_m": 500},
        }
        entry = ModelEntry(data)
        assert entry.quantizations == ["q4_k_m"]

    def test_default_fields(self):
        data = {"model_id": "min", "name": "Min", "architecture": "test", "params_m": 10, "vram_required_mb": {}}
        entry = ModelEntry(data)
        assert entry.hidden_size == 0
        assert entry.num_layers == 0
        assert entry.context_length == 4096
        assert entry.license == ""
        assert entry.huggingface_id == ""


class TestModelRegistry:
    def _make_registry(self, catalog_data=None):
        if catalog_data is None:
            catalog_data = CATALOG_DATA
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(catalog_data, f)
            f.flush()
            path = Path(f.name)
        reg = ModelRegistry(catalog_path=path)
        reg.load_local()
        return reg, path

    def test_load_local(self):
        reg, path = self._make_registry()
        assert len(reg._models) == 5
        assert "test-mini" in reg._models
        assert "test-small" in reg._models
        assert "test-mid" in reg._models
        assert "test-large" in reg._models
        Path(path).unlink()

    def test_load_local_missing_catalog_file(self):
        reg = ModelRegistry(catalog_path="/nonexistent/path/catalog.json")
        result = reg.load_local()
        if CATALOG_CACHE_PATH.exists():
            assert result is True
            assert len(reg._models) > 0
        else:
            assert result is False

    def test_get_existing(self):
        reg, path = self._make_registry()
        entry = reg.get("test-mini")
        assert entry is not None
        assert entry.model_id == "test-mini"
        Path(path).unlink()

    def test_get_missing(self):
        reg, path = self._make_registry()
        assert reg.get("nonexistent") is None
        Path(path).unlink()

    def test_list_models_all(self):
        reg, path = self._make_registry()
        models = reg.list_models()
        assert len(models) == 5
        assert models[0].params_m <= models[-1].params_m
        Path(path).unlink()

    def test_list_models_by_size_class(self):
        reg, path = self._make_registry()
        mini = reg.list_models(ModelSizeClass.MINI)
        assert len(mini) == 2
        assert all(m.size_class == ModelSizeClass.MINI for m in mini)
        small = reg.list_models(ModelSizeClass.SMALL)
        assert len(small) == 1
        Path(path).unlink()

    def test_models_for_vram(self):
        reg, path = self._make_registry()
        fits = reg.models_for_vram(5000, "q4_k_m")
        model_ids = [m.model_id for m in fits]
        assert "test-mini" in model_ids
        assert "test-small" in model_ids
        assert "test-mid" not in model_ids
        assert "test-large" not in model_ids
        Path(path).unlink()

    def test_models_for_vram_tiny(self):
        reg, path = self._make_registry()
        fits = reg.models_for_vram(2500, "q4_k_m")
        model_ids = [m.model_id for m in fits]
        assert "test-mini" in model_ids
        assert "test-no-size" in model_ids
        assert "test-small" not in model_ids
        Path(path).unlink()

    def test_models_for_vram_massive(self):
        reg, path = self._make_registry()
        fits = reg.models_for_vram(500000, "q4_k_m")
        assert len(fits) == 5
        Path(path).unlink()

    def test_voting_priority_no_votes(self):
        reg, path = self._make_registry()
        prioritized = reg.voting_priority()
        assert len(prioritized) == 5
        mini_first = [m for m in prioritized if m.size_class == ModelSizeClass.MINI]
        assert mini_first[0].params_m <= mini_first[-1].params_m
        Path(path).unlink()

    def test_voting_priority_with_votes(self):
        reg, path = self._make_registry()
        votes = {"test-large": 100.0, "test-mini": 50.0, "test-mid": 75.0, "test-small": 25.0}
        prioritized = reg.voting_priority(votes)
        assert prioritized[0].model_id == "test-large"
        assert prioritized[1].model_id == "test-mid"
        assert prioritized[2].model_id == "test-mini"
        assert prioritized[3].model_id == "test-small"
        Path(path).unlink()

    def test_size_classes_property(self):
        reg, path = self._make_registry()
        sc = reg.size_classes
        assert "mini" in sc
        assert sc["mini"]["min_params_m"] == 2
        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_fetch_remote_success(self):
        reg, path = self._make_registry()
        mock_data = dict(CATALOG_DATA)
        mock_data["models"] = [CATALOG_DATA["models"][0]]
        with patch("netai.inference.autoloader.aiohttp.ClientSession") as mock_session_cls:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_data)
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session
            result = await reg.fetch_remote()
            assert result is True
        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_fetch_remote_failure(self):
        reg, path = self._make_registry()
        with patch("netai.inference.autoloader.aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(side_effect=Exception("connection error"))
            mock_session_cls.return_value = mock_session
            result = await reg.fetch_remote()
            assert result is False
        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_refresh_skips_if_recent(self):
        reg, path = self._make_registry()
        reg._last_fetch = time.time()
        with patch.object(reg, "fetch_remote", new=AsyncMock()) as mock_fetch:
            result = await reg.refresh()
            mock_fetch.assert_not_called()
            assert result is True
        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_refresh_falls_back_to_local_on_failure(self):
        reg, path = self._make_registry()
        reg._last_fetch = 0
        reg._models = {}
        with patch.object(reg, "fetch_remote", new=AsyncMock(return_value=False)):
            result = await reg.refresh()
            assert result is True
            assert len(reg._models) > 0
        Path(path).unlink()

    def test_parse_catalog_invalid_entry_skipped(self):
        bad_data = {
            "models": [
                {"model_id": "good", "name": "Good", "architecture": "test", "params_m": 10, "vram_required_mb": {}},
                {"bad_entry": True},
            ]
        }
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(bad_data, f)
            f.flush()
            temp_path = Path(f.name)
        reg = ModelRegistry(catalog_path=temp_path)
        reg.load_local()
        assert len(reg._models) == 1
        temp_path.unlink()


class TestAutoLoader:
    def _make_loader(self, vram_mb=8000, nodes=1):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(CATALOG_DATA, f)
            f.flush()
            path = Path(f.name)
        reg = ModelRegistry(catalog_path=path)
        reg.load_local()
        loader = AutoLoader(reg, available_vram_mb=vram_mb, available_nodes=nodes)
        return loader, path

    def test_compute_load_plan_single_mini(self):
        loader, path = self._make_loader(vram_mb=3000)
        plan = loader.compute_load_plan()
        assert len(plan) >= 1
        assert plan[0]["model_id"] in ("test-mini", "test-no-size")
        path.unlink()

    def test_compute_load_plan_fits_multiple(self):
        loader, path = self._make_loader(vram_mb=8000)
        plan = loader.compute_load_plan()
        assert len(plan) >= 1
        total_vram = sum(p["vram_mb"] for p in plan)
        assert total_vram <= 8000 * 0.80
        path.unlink()

    def test_compute_load_plan_with_votes(self):
        loader, path = self._make_loader(vram_mb=8000)
        votes = {"test-small": 100.0, "test-mini": 50.0}
        plan = loader.compute_load_plan(votes=votes)
        if plan:
            assert plan[0]["model_id"] == "test-small"
        path.unlink()

    def test_compute_load_plan_force_models(self):
        loader, path = self._make_loader(vram_mb=5000)
        plan = loader.compute_load_plan(force_models=["test-mini"])
        assert len(plan) == 1
        assert plan[0]["model_id"] == "test-mini"
        path.unlink()

    def test_compute_load_plan_force_model_too_large(self):
        loader, path = self._make_loader(vram_mb=2000)
        plan = loader.compute_load_plan(force_models=["test-small"])
        assert len(plan) == 0
        path.unlink()

    def test_compute_load_plan_max_concurrent(self):
        loader, path = self._make_loader(vram_mb=500000)
        plan = loader.compute_load_plan()
        assert len(plan) <= loader.max_concurrent_models
        path.unlink()

    def test_compute_load_plan_pipeline_parallel(self):
        loader, path = self._make_loader(vram_mb=50000, nodes=4)
        loader.min_nodes_for_pipeline = 2
        plan = loader.compute_load_plan()
        assert len(plan) >= 1
        path.unlink()

    def test_compute_load_plan_no_pipeline_no_mid(self):
        loader, path = self._make_loader(vram_mb=50000, nodes=1)
        loader.min_nodes_for_pipeline = 2
        loader.max_concurrent_models = 10
        plan = loader.compute_load_plan()
        size_classes = [p["size_class"] for p in plan]
        assert "mid" not in size_classes or len(plan) == 0 or loader.available_nodes >= loader.min_nodes_for_pipeline
        path.unlink()

    def test_mark_loaded_and_unloaded(self):
        loader, path = self._make_loader(vram_mb=8000)
        loader.mark_loaded("test-mini", 1900)
        loaded = loader.get_loaded_models()
        assert len(loaded) == 1
        assert loaded[0]["model_id"] == "test-mini"

        loader.mark_unloaded("test-mini")
        loaded = loader.get_loaded_models()
        assert len(loaded) == 0
        path.unlink()

    def test_mark_pending_dedup(self):
        loader, path = self._make_loader(vram_mb=8000)
        loader.mark_pending("test-mini")
        loader.mark_pending("test-mini")
        assert "test-mini" in loader._pending_loads
        loader.mark_loaded("test-mini", 1900)
        assert "test-mini" not in loader._pending_loads
        path.unlink()

    def test_should_unload_overcommitted(self):
        loader, path = self._make_loader(vram_mb=2000)
        loader.mark_loaded("test-mini", 1900)
        loader.mark_loaded("test-no-size", 300)
        assert loader.should_unload("test-no-size") is True
        path.unlink()

    def test_should_not_unload_when_ok(self):
        loader, path = self._make_loader(vram_mb=10000)
        loader.mark_loaded("test-mini", 1900)
        assert loader.should_unload("test-mini") is False
        path.unlink()

    def test_should_unload_unknown_model(self):
        loader, path = self._make_loader(vram_mb=8000)
        assert loader.should_unload("nonexistent") is False
        path.unlink()

    def test_get_status(self):
        loader, path = self._make_loader(vram_mb=8000)
        loader.mark_loaded("test-mini", 1900)
        status = loader.get_status()
        assert "available_vram_mb" in status
        assert "loaded_models" in status
        assert "recommended_loads" in status
        assert "preferred_quant" in status
        assert "catalog_size" in status
        assert status["catalog_size"] == 5
        assert len(status["loaded_models"]) == 1
        path.unlink()

    def test_update_resources(self):
        loader, path = self._make_loader(vram_mb=8000)
        loader.update_resources(16000, available_nodes=2)
        assert loader.available_vram_mb == 16000
        assert loader.available_nodes == 2
        plan = loader.compute_load_plan()
        assert len(plan) >= 1
        path.unlink()

    def test_load_plan_skips_already_loaded(self):
        loader, path = self._make_loader(vram_mb=8000)
        loader.mark_loaded("test-mini", 1900)
        plan = loader.compute_load_plan()
        model_ids = [p["model_id"] for p in plan]
        assert "test-mini" not in model_ids
        path.unlink()

    def test_load_plan_skips_pending(self):
        loader, path = self._make_loader(vram_mb=8000)
        loader.mark_pending("test-mini")
        plan = loader.compute_load_plan()
        model_ids = [p["model_id"] for p in plan]
        assert "test-mini" not in model_ids
        path.unlink()

    def test_load_plan_falls_back_to_mini_when_nothing_fits(self):
        loader, path = self._make_loader(vram_mb=500)
        loader.max_concurrent_models = 5
        plan = loader.compute_load_plan()
        model_ids = [p["model_id"] for p in plan]
        assert "test-no-size" in model_ids
        path.unlink()

    def test_quant_preference(self):
        loader, path = self._make_loader(vram_mb=8000)
        loader.preferred_quant = "q5_k_m"
        plan = loader.compute_load_plan()
        for p in plan:
            assert p["quant"] == "q5_k_m"
        path.unlink()

    def test_load_history(self):
        loader, path = self._make_loader(vram_mb=8000)
        loader.mark_loaded("test-mini", 1900)
        loader.mark_loaded("test-small", 4100)
        assert len(loader._load_history) == 2
        assert loader._load_history[0][0] == "test-mini"
        assert loader._load_history[1][0] == "test-small"
        path.unlink()

    def test_compute_load_plan_returns_config(self):
        loader, path = self._make_loader(vram_mb=5000)
        plan = loader.compute_load_plan(force_models=["test-mini"])
        assert len(plan) == 1
        config = plan[0]["config"]
        assert "model_id" in config
        assert "hidden_size" in config
        assert "num_layers" in config
        assert "quantization" in config
        path.unlink()

    def test_compute_load_plan_shard_calculation(self):
        loader, path = self._make_loader(vram_mb=8000)
        plan = loader.compute_load_plan(force_models=["test-mini"])
        if plan:
            assert plan[0]["shards"] >= 1
        path.unlink()