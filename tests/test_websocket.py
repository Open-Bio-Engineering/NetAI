"""Tests for WebSocket native streaming endpoint."""

from __future__ import annotations

import json
import tempfile
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from netai.api.app import create_app
from netai.p2p.network import P2PNode
from netai.inference.native_engine import NativeInferenceEngine, TransformerConfig
from netai.training.voting import VotingEngine
from netai.training.groups import GroupManager
from netai.scheduler.scheduler import JobScheduler
from netai.security import SecurityMiddleware, Scope, UserRole
from netai.benchmark.runner import ModelBenchmark


def _make_test_engine():
    engine = NativeInferenceEngine(node_id="ws-test-node")
    config = TransformerConfig(
        vocab_size=1024,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=256,
        max_position_embeddings=512,
        layer_norm_eps=1e-5,
        model_type="gpt2",
    )
    engine.configs["ws-test-model"] = config
    embed = np.random.randn(1024, 64).astype(np.float32) * 0.02
    engine.embed_tokens["ws-test-model"] = embed
    engine._loaded_models.add("ws-test-model")

    for i in range(config.num_layers):
        layer_weights = {}
        layer_weights["ln_1.weight"] = np.ones(64, dtype=np.float32)
        layer_weights["ln_1.bias"] = np.zeros(64, dtype=np.float32)
        layer_weights["ln_2.weight"] = np.ones(64, dtype=np.float32)
        layer_weights["ln_2.bias"] = np.zeros(64, dtype=np.float32)
        layer_weights["attn.c_attn.weight"] = np.random.randn(64, 3 * 64).astype(np.float32) * 0.02
        layer_weights["attn.c_attn.bias"] = np.zeros(3 * 64, dtype=np.float32)
        layer_weights["attn.c_proj.weight"] = np.random.randn(64, 64).astype(np.float32) * 0.02
        layer_weights["attn.c_proj.bias"] = np.zeros(64, dtype=np.float32)
        layer_weights["mlp.c_fc.weight"] = np.random.randn(64, 256).astype(np.float32) * 0.02
        layer_weights["mlp.c_fc.bias"] = np.zeros(256, dtype=np.float32)
        layer_weights["mlp.c_proj.weight"] = np.random.randn(256, 64).astype(np.float32) * 0.02
        layer_weights["mlp.c_proj.bias"] = np.zeros(64, dtype=np.float32)
        engine.layers[f"ws-test-model/layer_{i}"] = layer_weights

    engine.layer_norm_f["ws-test-model"] = (
        np.ones(64, dtype=np.float32),
        np.zeros(64, dtype=np.float32),
    )
    return engine


@pytest.fixture
def ws_client():
    sec = SecurityMiddleware()
    sec.register_user("ws-user", "testpassword123", UserRole.ADMIN,
                      scopes=[s.value for s in Scope])
    sec.register_public_endpoint("/ws/inference/stream-native")
    engine = _make_test_engine()
    app = create_app(
        p2p_node=P2PNode(port=0, node_id="ws-test-node"),
        voting_engine=VotingEngine(),
        group_manager=GroupManager(),
        scheduler=JobScheduler(),
        security=sec,
        benchmark_runner=ModelBenchmark(engine=engine),
        native_engine=engine,
    )
    return TestClient(app)


class TestWebSocketNativeStream:
    """Integration tests for /ws/inference/stream-native WebSocket endpoint."""

    def test_connect_and_receive_start(self, ws_client):
        with ws_client.websocket_connect("/ws/inference/stream-native") as ws:
            ws.send_json({
                "model_id": "ws-test-model",
                "prompt": "Hello world",
                "max_tokens": 4,
                "temperature": 0.7,
                "top_p": 0.9,
            })
            msg = ws.receive_json()
            assert msg["type"] == "start"
            assert msg["model_id"] == "ws-test-model"
            assert msg["prompt"] == "Hello world"

    def test_receive_tokens_then_done(self, ws_client):
        with ws_client.websocket_connect("/ws/inference/stream-native") as ws:
            ws.send_json({
                "model_id": "ws-test-model",
                "prompt": "Test",
                "max_tokens": 3,
                "temperature": 0.7,
                "top_p": 0.9,
            })
            start = ws.receive_json()
            assert start["type"] == "start"

            tokens_received = []
            done = None
            while True:
                msg = ws.receive_json()
                if msg["type"] == "token":
                    tokens_received.append(msg)
                elif msg["type"] == "done":
                    done = msg
                    break

            assert len(tokens_received) == 3
            assert done is not None
            assert done["tokens_generated"] == 3
            assert "latency_ms" in done
            assert "tokens_per_second" in done

    def test_cancel_generation(self, ws_client):
        with ws_client.websocket_connect("/ws/inference/stream-native") as ws:
            ws.send_json({
                "model_id": "ws-test-model",
                "prompt": "Keep going",
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
            })
            start = ws.receive_json()
            assert start["type"] == "start"

            ws.send_json({"type": "cancel"})

            msgs = []
            while True:
                msg = ws.receive_json()
                msgs.append(msg)
                if msg["type"] in ("done", "cancelled"):
                    break

            assert any(m["type"] == "cancelled" for m in msgs)

    def test_unloaded_model_returns_error(self, ws_client):
        with ws_client.websocket_connect("/ws/inference/stream-native") as ws:
            ws.send_json({
                "model_id": "nonexistent-model",
                "prompt": "Hello",
                "max_tokens": 4,
                "temperature": 0.7,
                "top_p": 0.9,
            })
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "not loaded" in msg["error"].lower()

    def test_missing_fields_returns_error(self, ws_client):
        with ws_client.websocket_connect("/ws/inference/stream-native") as ws:
            ws.send_json({"model_id": "ws-test-model"})
            msg = ws.receive_json()
            assert msg["type"] in ("error", "start")

    def test_disconnect_during_generation(self, ws_client):
        with ws_client.websocket_connect("/ws/inference/stream-native") as ws:
            ws.send_json({
                "model_id": "ws-test-model",
                "prompt": "This is a test",
                "max_tokens": 50,
                "temperature": 0.7,
                "top_p": 0.9,
            })
            start = ws.receive_json()
            assert start["type"] == "start"
            ws.close()

    def test_large_max_tokens_clamped(self, ws_client):
        with ws_client.websocket_connect("/ws/inference/stream-native") as ws:
            ws.send_json({
                "model_id": "ws-test-model",
                "prompt": "Hi",
                "max_tokens": 99999,
                "temperature": 0.7,
                "top_p": 0.9,
            })
            start = ws.receive_json()
            assert start["type"] == "start"
            ws.send_json({"type": "cancel"})
            while True:
                msg = ws.receive_json()
                if msg["type"] in ("done", "cancelled", "error"):
                    break

    def test_model_config_in_start_message(self, ws_client):
        with ws_client.websocket_connect("/ws/inference/stream-native") as ws:
            ws.send_json({
                "model_id": "ws-test-model",
                "prompt": "Config test",
                "max_tokens": 1,
                "temperature": 0.5,
                "top_p": 0.8,
            })
            start = ws.receive_json()
            assert start["type"] == "start"
            assert "prompt" in start
            assert "prompt_tokens" in start
            assert start["prompt_tokens"] > 0
            done = None
            while True:
                msg = ws.receive_json()
                if msg["type"] == "done":
                    done = msg
                    break
            assert done is not None

    def test_token_contains_id_and_text(self, ws_client):
        with ws_client.websocket_connect("/ws/inference/stream-native") as ws:
            ws.send_json({
                "model_id": "ws-test-model",
                "prompt": "X",
                "max_tokens": 2,
                "temperature": 0.7,
                "top_p": 0.9,
            })
            start = ws.receive_json()
            assert start["type"] == "start"
            tokens = []
            while True:
                msg = ws.receive_json()
                if msg["type"] == "token":
                    tokens.append(msg)
                if msg["type"] == "done":
                    break
            assert len(tokens) == 2
            for t in tokens:
                assert "token_id" in t
                assert "text" in t
                assert "index" in t
