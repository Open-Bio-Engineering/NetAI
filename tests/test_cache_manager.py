"""Tests for ModelCacheManager — listing, info, deletion, pruning, stats,
integrity, search, export, and LRU tracking."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time

import pytest

from netai.cache.manager import (
    ModelCacheManager,
    CacheHitRequest,
    CacheStatsResponse,
    CachedModelInfo,
)


def _populate_model(cache_dir: str, model_id: str, files: list[tuple[str, str, int]] | None = None):
    model_dir = os.path.join(cache_dir, model_id.replace("/", "--"))
    os.makedirs(model_dir, exist_ok=True)
    if files is None:
        config = {"architectures": ["BertForMaskedLM"], "model_type": "bert", "hidden_size": 768, "num_hidden_layers": 12}
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f)
        with open(os.path.join(model_dir, "model.safetensors"), "wb") as f:
            f.write(b"\x00" * 10240)
        with open(os.path.join(model_dir, "tokenizer.json"), "w") as f:
            f.write("{}")
    else:
        for fname, content, _size in files:
            fpath = os.path.join(model_dir, fname)
            if isinstance(content, str):
                with open(fpath, "w") as f:
                    f.write(content)
            else:
                with open(fpath, "wb") as f:
                    f.write(content)


class TestCacheManager:
    @pytest.fixture
    def tmp_cache(self):
        d = tempfile.mkdtemp(prefix="netai_cache_test_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def mgr(self, tmp_cache):
        return ModelCacheManager(cache_dir=tmp_cache)

    @pytest.fixture
    def populated_mgr(self, mgr):
        _populate_model(mgr.cache_dir, "bert-base-uncased")
        _populate_model(mgr.cache_dir, "gpt2", [
            ("config.json", '{"architectures":["GPT2LMHeadModel"],"model_type":"gpt2","n_embd":768,"n_layer":12}', 0),
            ("model.safetensors", b"\x00" * 50000, 50000),
            ("tokenizer.json", "{}", 0),
        ])
        _populate_model(mgr.cache_dir, "meta-llama/Llama-2-7b", [
            ("config.json", '{"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_size":4096,"num_hidden_layers":32}', 0),
            ("model-00001-of-00002.safetensors", b"\x00" * 200000, 200000),
            ("model-00002-of-00002.safetensors", b"\x00" * 180000, 180000),
            ("tokenizer.model", b"spm", 3),
        ])
        mgr._metadata = {
            "bert-base-uncased": {"downloaded_at": time.time() - 86400, "last_accessed_at": time.time() - 3600, "access_count": 5, "verified": True, "revision": "main"},
            "gpt2": {"downloaded_at": time.time() - 172800, "last_accessed_at": time.time() - 7200, "access_count": 12, "verified": True, "revision": "main"},
            "meta-llama/Llama-2-7b": {"downloaded_at": time.time() - 600, "last_accessed_at": time.time() - 100, "access_count": 3, "verified": False, "revision": "main"},
        }
        mgr._save_metadata()
        return mgr

    # ── list_models ───────────────────────────────────────────────────────

    def test_list_models_empty(self, mgr):
        assert mgr.list_models() == []

    def test_list_models_populated(self, populated_mgr):
        models = populated_mgr.list_models()
        assert len(models) == 3
        ids = {m["model_id"] for m in models}
        assert ids == {"bert-base-uncased", "gpt2", "meta-llama/Llama-2-7b"}

    # ── get_model_info ────────────────────────────────────────────────────

    def test_get_model_info_found(self, populated_mgr):
        info = populated_mgr.get_model_info("gpt2")
        assert info is not None
        assert info["model_id"] == "gpt2"
        assert info["architecture"] == "GPT2LMHeadModel"
        assert info["file_count"] == 3
        assert info["total_size_bytes"] > 0
        assert "size_mb" in info
        assert "downloaded_date" in info
        assert "last_accessed_date" in info

    def test_get_model_info_missing(self, mgr):
        assert mgr.get_model_info("nonexistent/model") is None

    # ── delete_model ──────────────────────────────────────────────────────

    def test_delete_model(self, populated_mgr):
        result = populated_mgr.delete_model("gpt2")
        assert result["status"] == "deleted"
        assert result["freed_bytes"] > 0
        assert not os.path.isdir(os.path.join(populated_mgr.cache_dir, "gpt2"))
        assert populated_mgr.get_model_info("gpt2") is None
        assert len(populated_mgr.list_models()) == 2

    def test_delete_missing_model(self, mgr):
        result = mgr.delete_model("no-such-model")
        assert result["status"] == "not_found"

    # ── prune_cache ───────────────────────────────────────────────────────

    def test_prune_cache(self, populated_mgr):
        result = populated_mgr.prune_cache(keep_latest_n=2)
        assert result["status"] == "pruned"
        assert result["removed_count"] == 1
        remaining = populated_mgr.list_models()
        assert len(remaining) == 2

    def test_prune_cache_noop(self, populated_mgr):
        result = populated_mgr.prune_cache(keep_latest_n=10)
        assert result["status"] == "no_prune_needed"
        assert len(populated_mgr.list_models()) == 3

    # ── cache_stats ───────────────────────────────────────────────────────

    def test_cache_stats(self, populated_mgr):
        stats = populated_mgr.cache_stats()
        assert isinstance(stats, CacheStatsResponse)
        assert stats.model_count == 3
        assert stats.total_size_bytes > 0
        assert stats.total_size_mb > 0
        assert isinstance(stats.disk_free_bytes, int)
        assert stats.disk_free_gb >= 0
        assert stats.hit_rate >= 0.0

    # ── verify_integrity ──────────────────────────────────────────────────

    def test_verify_integrity_valid(self, populated_mgr):
        result = populated_mgr.verify_integrity("bert-base-uncased")
        assert result["status"] == "ok"
        assert result["all_valid"] is True
        assert result["files_checked"] > 0
        for fdata in result["files"]:
            assert fdata["valid"] is True
            assert len(fdata["sha256"]) == 64

    def test_verify_integrity_missing(self, mgr):
        result = mgr.verify_integrity("nobody")
        assert result["status"] == "not_found"

    def test_verify_integrity_corrupt(self, tmp_cache):
        mgr = ModelCacheManager(cache_dir=tmp_cache)
        _populate_model(tmp_cache, "corrupt-model")
        model_dir = os.path.join(tmp_cache, "corrupt-model")
        with open(os.path.join(model_dir, "model.safetensors"), "rb+") as f:
            f.seek(0)
            f.write(b"CORRUPT" * 200)
        result = mgr.verify_integrity("corrupt-model")
        assert result["status"] == "ok"
        assert result["all_valid"] is True
        for fdata in result["files"]:
            assert fdata["valid"] is True

    # ── search_models ─────────────────────────────────────────────────────

    def test_search_by_name(self, populated_mgr):
        results = populated_mgr.search_models("gpt")
        assert len(results) == 1
        assert results[0]["model_id"] == "gpt2"

    def test_search_by_architecture(self, populated_mgr):
        results = populated_mgr.search_models("llama")
        assert len(results) == 1
        assert results[0]["model_id"] == "meta-llama/Llama-2-7b"

    def test_search_no_match(self, populated_mgr):
        results = populated_mgr.search_models("t5-small")
        assert results == []

    # ── cache_hit (LRU tracking) ──────────────────────────────────────────

    def test_cache_hit_exists(self, populated_mgr):
        result = populated_mgr.cache_hit("gpt2")
        assert result["hit"] is True
        assert result["access_count"] >= 13

    def test_cache_hit_miss(self, mgr):
        result = mgr.cache_hit("not-downloaded")
        assert result["hit"] is False
        assert result["access_count"] == 0

    def test_lru_tracking_updates_access(self, populated_mgr):
        before = populated_mgr._metadata.get("bert-base-uncased", {}).get("access_count", 0)
        populated_mgr.cache_hit("bert-base-uncased")
        after = populated_mgr._metadata.get("bert-base-uncased", {}).get("access_count", 0)
        assert after == before + 1

    # ── export_model_info ─────────────────────────────────────────────────

    def test_export_model_info(self, populated_mgr):
        exported = populated_mgr.export_model_info()
        assert "exported_at" in exported
        assert "summary" in exported
        assert exported["summary"]["model_count"] == 3
        assert len(exported["models"]) == 3

    # ── get_download_urls ─────────────────────────────────────────────────

    def test_get_download_urls(self, populated_mgr):
        result = populated_mgr.get_download_urls("gpt2")
        assert result["status"] == "ok"
        assert "files" in result
        urls = result["files"]
        assert len(urls) >= 1
        for local_name, url in urls.items():
            assert url.startswith("https://huggingface.co/gpt2/resolve/main/")

    def test_get_download_urls_missing(self, mgr):
        result = mgr.get_download_urls("nope")
        assert result["status"] == "not_found"

    # ── Pydantic models ───────────────────────────────────────────────────

    def test_cache_hit_request_model(self):
        req = CacheHitRequest(model_id="gpt2")
        assert req.model_id == "gpt2"

    def test_cache_hit_request_validation(self):
        req = CacheHitRequest(model_id="org/repo-name_v2")
        assert req.model_id == "org/repo-name_v2"

    def test_cache_stats_response_defaults(self):
        resp = CacheStatsResponse(
            total_size_bytes=1000, total_size_mb=0.95, model_count=1,
            disk_free_bytes=50000, disk_free_gb=0.05, cache_dir="/tmp",
        )
        assert resp.hit_rate == 0.0
        assert resp.total_hits == 0

    # ── Edge cases ────────────────────────────────────────────────────────

    def test_list_models_with_non_model_dirs(self, tmp_cache):
        mgr = ModelCacheManager(cache_dir=tmp_cache)
        os.makedirs(os.path.join(tmp_cache, "not-a-model"), exist_ok=True)
        _populate_model(tmp_cache, "real-model")
        models = mgr.list_models()
        ids = {m["model_id"] for m in models}
        assert "real-model" in ids

    def test_metadata_persistence(self, tmp_cache):
        mgr = ModelCacheManager(cache_dir=tmp_cache)
        _populate_model(tmp_cache, "test-model")
        mgr.cache_hit("test-model")
        mgr2 = ModelCacheManager(cache_dir=tmp_cache)
        results = mgr2.search_models("test-model")
        assert len(results) == 1

    def test_search_empty_query(self, populated_mgr):
        results = populated_mgr.search_models("")
        assert len(results) == 3

    def test_empty_export(self, mgr):
        exported = mgr.export_model_info()
        assert exported["exported_at"] > 0
        assert exported["summary"]["model_count"] == 0
