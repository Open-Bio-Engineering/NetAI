"""Comprehensive security tests - auth, rate limiting, input validation, gradient integrity, model provenance."""

import asyncio
import numpy as np
import pytest
from netai.security import (
    SecurityMiddleware, AuthDependency, Scope, UserRole, AuditLogger,
    RateLimiter, InputValidator, GradientIntegrityChecker, ModelProvenance,
)
from netai.crypto.identity import NodeIdentity


class TestSecurityMiddleware:
    def test_init(self):
        sec = SecurityMiddleware()
        assert sec.node_identity is not None
        assert "admin" in sec.users

    def test_register_user(self):
        sec = SecurityMiddleware()
        user = sec.register_user("alice", "password123")
        assert user.user_id == "alice"
        assert user.role == UserRole.USER
        assert Scope.READ.value in user.scopes

    def test_register_user_duplicate(self):
        sec = SecurityMiddleware()
        sec.register_user("bob", "password123")
        with pytest.raises(ValueError):
            sec.register_user("bob", "password123")

    def test_register_user_short_password(self):
        sec = SecurityMiddleware()
        with pytest.raises(ValueError):
            sec.register_user("charlie", "short")

    def test_authenticate_password(self):
        sec = SecurityMiddleware()
        sec.register_user("dave", "goodpassword1")
        user = sec.authenticate_password("dave", "goodpassword1")
        assert user is not None
        assert user.user_id == "dave"

    def test_authenticate_wrong_password(self):
        sec = SecurityMiddleware()
        sec.register_user("eve", "mypassword1")
        user = sec.authenticate_password("eve", "wrongpassword")
        assert user is None

    def test_authenticate_nonexistent_user(self):
        sec = SecurityMiddleware()
        user = sec.authenticate_password("nobody", "password")
        assert user is None

    def test_create_token(self):
        sec = SecurityMiddleware()
        sec.register_user("frank", "password123")
        token = sec.create_token("frank", ["read", "write"], 1.0)
        assert token.token
        assert "read" in token.scopes
        assert "write" in token.scopes

    def test_create_token_nonexistent_user(self):
        sec = SecurityMiddleware()
        with pytest.raises(ValueError):
            sec.create_token("nobody", ["read"])

    def test_verify_token(self):
        sec = SecurityMiddleware()
        sec.register_user("grace", "password123")
        token = sec.create_token("grace", ["read"], 24.0)
        verified = sec.verify_token(token.token)
        assert verified is not None
        assert verified.user_id == "grace"

    def test_verify_expired_token(self):
        sec = SecurityMiddleware()
        sec.register_user("heidi", "password123")
        token = sec.create_token("heidi", ["read"], 0.000001)
        import time
        time.sleep(0.01)
        verified = sec.verify_token(token.token)
        assert verified is None

    def test_verify_invalid_token(self):
        sec = SecurityMiddleware()
        verified = sec.verify_token("invalid_token_string")
        assert verified is None

    def test_create_api_key(self):
        sec = SecurityMiddleware()
        sec.register_user("ivan", "password123")
        key = sec.create_api_key("ivan", "my-key", ["read", "write"])
        assert key.key.startswith("nx_")
        assert key.name == "my-key"

    def test_verify_api_key(self):
        sec = SecurityMiddleware()
        sec.register_user("judy", "password123")
        key = sec.create_api_key("judy", "test", ["read"])
        verified = sec.verify_api_key(key.key)
        assert verified is not None
        assert verified.user_id == "judy"

    def test_verify_invalid_api_key(self):
        sec = SecurityMiddleware()
        verified = sec.verify_api_key("nx_invalid_key")
        assert verified is None

    def test_revoke_token(self):
        sec = SecurityMiddleware()
        sec.register_user("karl", "password123")
        token = sec.create_token("karl", ["read"], 1.0)
        assert sec.revoke_token(token.token)
        assert sec.verify_token(token.token) is None

    def test_revoke_api_key(self):
        sec = SecurityMiddleware()
        sec.register_user("leon", "password123")
        key = sec.create_api_key("leon", "test")
        assert sec.revoke_api_key(key.key)
        assert sec.verify_api_key(key.key) is None

    def test_check_permission_admin(self):
        sec = SecurityMiddleware()
        sec.register_user("admin2", "password123", UserRole.ADMIN, ["admin", "read", "write"])
        token = sec.create_token("admin2", ["admin", "read", "write"])
        assert sec.check_permission(token, Scope.ADMIN.value)

    def test_check_permission_user_limited(self):
        sec = SecurityMiddleware()
        sec.register_user("limited", "password123", UserRole.USER, ["read"])
        token = sec.create_token("limited", ["read"])
        assert sec.check_permission(token, Scope.READ.value)
        assert not sec.check_permission(token, Scope.ADMIN.value)

    def test_sign_and_verify_p2p_message(self):
        sec = SecurityMiddleware()
        msg = {"msg_type": "gradient", "payload": {"step": 1}}
        signed = sec.sign_p2p_message(msg)
        assert "signature" in signed
        assert signed["sender_id"] == sec.node_identity.node_id
        assert sec.verify_p2p_message(signed, sec.node_identity.verification_key)

    def test_verify_tampered_p2p_message(self):
        sec = SecurityMiddleware()
        msg = {"msg_type": "gradient", "payload": {"step": 1}}
        signed = sec.sign_p2p_message(msg)
        signed["payload"]["step"] = 999
        assert not sec.verify_p2p_message(signed, sec.node_identity.verification_key)

    def test_verify_unsigned_p2p_message(self):
        sec = SecurityMiddleware()
        msg = {"msg_type": "gradient", "payload": {"step": 1}}
        assert not sec.verify_p2p_message(msg)

    def test_public_endpoints(self):
        sec = SecurityMiddleware()
        assert sec.is_public("/")
        assert sec.is_public("/api/demo")
        assert sec.is_public("/api/status")
        assert sec.is_public("/api/metrics")
        assert sec.is_public("/api/auth/login")
        assert not sec.is_public("/api/training/submit")

    def test_get_security_status(self):
        sec = SecurityMiddleware()
        sec.register_user("test", "password123")
        status = sec.get_security_status()
        assert "node_id" in status
        assert status["users_registered"] >= 2
        assert "audit_stats" in status

    def test_disabled_user(self):
        sec = SecurityMiddleware()
        sec.register_user("disabled1", "password123")
        token = sec.create_token("disabled1", ["read"], 24.0)
        assert token is not None
        verified = sec.verify_token(token.token)
        assert verified is not None
        sec.users["disabled1"].disabled = True
        verified = sec.verify_token(token.token)
        assert verified is None


class TestAuditLogger:
    def test_log_event(self):
        al = AuditLogger()
        al.log("auth_success", user_id="alice", ip_address="1.2.3.4")
        events = al.get_recent(10)
        assert len(events) == 1
        assert events[0]["event_type"] == "auth_success"

    def test_alert_on_high_risk(self):
        al = AuditLogger()
        al.log("tamper_detected", user_id="bob", risk_score=0.9)
        events = al.get_alerts()
        assert len(events) == 1

    def test_brute_force_detection(self):
        al = AuditLogger()
        for _ in range(11):
            al.log("auth_failure", ip_address="10.0.0.1")
        assert al.check_brute_force("10.0.0.1", window=300, threshold=10)

    def test_no_brute_force_below_threshold(self):
        al = AuditLogger()
        for _ in range(5):
            al.log("auth_failure", ip_address="10.0.0.2")
        assert not al.check_brute_force("10.0.0.2", window=300, threshold=10)

    def test_get_stats(self):
        al = AuditLogger()
        al.log("auth_success", user_id="u1")
        al.log("auth_success", user_id="u2")
        al.log("auth_failure", user_id="u1")
        stats = al.get_stats()
        assert stats["total_events"] == 3
        assert stats["event_counts"]["auth_success"] == 2
        assert stats["event_counts"]["auth_failure"] == 1


class TestRateLimiter:
    def test_basic_rate_limiting(self):
        rl = RateLimiter()
        rl.set_rule("test", max_requests=3, window_seconds=60.0, burst=10)
        for _ in range(3):
            ok, _ = asyncio.run(rl.check("key1", "test"))
            assert ok
        ok, info = asyncio.run(rl.check("key1", "test"))
        assert not ok

    def test_different_keys_independent(self):
        rl = RateLimiter()
        rl.set_rule("test", max_requests=2, window_seconds=60.0, burst=10)
        ok1, _ = asyncio.run(rl.check("keyA", "test"))
        ok2, _ = asyncio.run(rl.check("keyB", "test"))
        assert ok1 and ok2

    def test_block(self):
        rl = RateLimiter()
        rl.block("bad_key", 3600)
        assert rl.is_blocked("bad_key")

    def test_block_expiry(self):
        rl = RateLimiter()
        rl.block("temp_key", 0.001)
        import time
        time.sleep(0.02)
        assert not rl.is_blocked("temp_key")


class TestInputValidator:
    def test_sanitize_string(self):
        assert InputValidator.validate_prompt("Hello world") == "Hello world"

    def test_sanitize_dangerous_patterns(self):
        with pytest.raises(ValueError):
            InputValidator.validate_prompt("<script>alert('xss')</script>")
        with pytest.raises(ValueError):
            InputValidator.validate_prompt("{{template_injection}}")
        with pytest.raises(ValueError):
            InputValidator.validate_prompt("${jndi:ldap://evil}")

    def test_validate_model_name(self):
        assert InputValidator.validate_model_name("gpt2-small") == "gpt2-small"

    def test_validate_model_name_invalid_chars(self):
        with pytest.raises(ValueError):
            InputValidator.validate_model_name("model with spaces")
        with pytest.raises(ValueError):
            InputValidator.validate_model_name("model;drop table")

    def test_validate_model_name_empty(self):
        with pytest.raises(ValueError):
            InputValidator.validate_model_name("")

    def test_validate_positive_int(self):
        assert InputValidator.validate_positive_int(8, "batch_size") == 8
        with pytest.raises(ValueError):
            InputValidator.validate_positive_int(0, "batch_size")
        with pytest.raises(ValueError):
            InputValidator.validate_positive_int(-1, "batch_size")

    def test_validate_device(self):
        assert InputValidator.validate_device("cuda") == "cuda"
        with pytest.raises(ValueError):
            InputValidator.validate_device("invalid_device")

    def test_validate_architecture(self):
        assert InputValidator.validate_architecture("transformer") == "transformer"
        with pytest.raises(ValueError):
            InputValidator.validate_architecture("unknown_arch")

    def test_validate_visibility(self):
        assert InputValidator.validate_visibility("public") == "public"
        with pytest.raises(ValueError):
            InputValidator.validate_visibility("hidden")

    def test_validate_user_id(self):
        assert InputValidator.validate_user_id("alice") == "alice"
        with pytest.raises(ValueError):
            InputValidator.validate_user_id("")

    def test_prompt_too_long(self):
        with pytest.raises(ValueError):
            InputValidator.validate_prompt("x" * 40000)

    def test_gradient_data_validation(self):
        data = {"layer_0": {"compressed": {"method": "topk", "values": [1.0, 2.0]}}}
        size = InputValidator.validate_gradient_data(data, max_size_mb=100)
        assert size < 1.0

    def test_gradient_data_invalid_method(self):
        data = {"layer_0": {"compressed": {"method": "evil", "values": []}}}
        with pytest.raises(ValueError):
            InputValidator.validate_gradient_data(data)


class TestGradientIntegrityChecker:
    def test_compute_gradient_hash(self):
        gic = GradientIntegrityChecker()
        grad = np.ones((4, 4), dtype=np.float32)
        h = gic.compute_gradient_hash(grad, "layer_0", "job-1", 1, "node-1")
        assert len(h) == 32

    def test_verify_valid_gradient(self):
        gic = GradientIntegrityChecker()
        grad = np.random.randn(4, 4).astype(np.float32) * 0.1
        ok, msg = gic.verify_gradient(grad, "layer_0", "job-1", 1, "node-1")
        assert ok
        assert msg == "ok"

    def test_verify_nan_gradient(self):
        gic = GradientIntegrityChecker()
        grad = np.full((4, 4), float("nan"), dtype=np.float32)
        ok, msg = gic.verify_gradient(grad, "layer_0", "job-1", 1, "node-1")
        assert not ok
        assert "invalid norm" in msg

    def test_verify_inf_gradient(self):
        gic = GradientIntegrityChecker()
        grad = np.full((4, 4), float("inf"), dtype=np.float32)
        ok, msg = gic.verify_gradient(grad, "layer_0", "job-1", 1, "node-1")
        assert not ok

    def test_verify_oversized_gradient(self):
        gic = GradientIntegrityChecker()
        gic.max_gradient_norm = 1.0
        grad = np.ones((100, 100), dtype=np.float32) * 10.0
        ok, msg = gic.verify_gradient(grad, "layer_0", "job-1", 1, "node-1")
        assert not ok
        assert "exceeds max" in msg

    def test_verify_signed_gradient(self):
        gic = GradientIntegrityChecker()
        signer = NodeIdentity.generate("signer-node")
        grad = np.random.randn(4, 4).astype(np.float32) * 0.1
        h = gic.compute_gradient_hash(grad, "layer_0", "job-1", 1, "node-1")
        sig = signer.sign(h.encode())
        ok, msg = gic.verify_node_gradient(grad, "layer_0", "job-1", 1, "node-1",
                                            signature=sig, signer=signer)
        assert ok

    def test_verify_tampered_signed_gradient(self):
        gic = GradientIntegrityChecker()
        signer = NodeIdentity.generate("signer-node")
        grad = np.random.randn(4, 4).astype(np.float32) * 0.1
        h = gic.compute_gradient_hash(grad, "layer_0", "job-1", 1, "node-1")
        sig = signer.sign(h.encode())
        tampered = grad * 100.0
        ok, msg = gic.verify_node_gradient(tampered, "layer_0", "job-1", 1, "node-1",
                                            signature=sig, signer=signer)
        assert not ok
        assert "signature" in msg

    def test_byzantine_aggregate(self):
        gic = GradientIntegrityChecker()
        honest1 = np.ones((4, 4), dtype=np.float32) * 0.1
        honest2 = np.ones((4, 4), dtype=np.float32) * 0.2
        evil = np.ones((4, 4), dtype=np.float32) * 1000.0
        gradients = {
            "node-1": {"layer_0": honest1},
            "node-2": {"layer_0": honest2},
            "evil-node": {"layer_0": evil},
        }
        result = gic.byzantine_aggregate(gradients, "job-1", 1)
        assert "layer_0" in result
        avg_val = float(result["layer_0"].mean())
        assert avg_val < 10.0

    def test_byzantine_aggregate_empty(self):
        gic = GradientIntegrityChecker()
        result = gic.byzantine_aggregate({})
        assert result == {}

    def test_byzantine_trust_scores(self):
        gic = GradientIntegrityChecker()
        good = np.ones((4, 4), dtype=np.float32) * 0.1
        bad = np.full((4, 4), float("nan"), dtype=np.float32)
        gradients1 = {"node-1": {"layer_0": good}, "node-2": {"layer_0": good}}
        gic.byzantine_aggregate(gradients1, "job-1", 1)
        scores = gic.get_node_trust_scores()
        assert "node-1" in scores
        assert scores["node-1"]["trust_score"] >= 0.0

    def test_get_status(self):
        gic = GradientIntegrityChecker()
        status = gic.get_status()
        assert "max_gradient_norm" in status
        assert "node_trust_scores" in status


class TestModelProvenance:
    def test_register_model(self):
        mp = ModelProvenance()
        prov = mp.register_model("gpt2-small", "huggingface", "alice")
        assert prov["model_id"] == "gpt2-small"
        assert prov["owner_id"] == "alice"
        assert "provenance_hash" in prov

    def test_register_model_signed(self):
        mp = ModelProvenance()
        signer = NodeIdentity.generate("signer")
        prov = mp.register_model("llama-7b", "huggingface", "bob", node_identity=signer)
        assert "signature" in prov
        assert prov["signer_node_id"] == "signer"

    def test_verify_model_signature(self):
        mp = ModelProvenance()
        signer = NodeIdentity.generate("verifier")
        mp.register_model("mistral-7b", "local", "carol", node_identity=signer)
        ok, msg = mp.verify_model_signature("mistral-7b")
        assert ok

    def test_verify_model_no_signature(self):
        mp = ModelProvenance()
        mp.register_model("unsig-model", "local", "dave")
        ok, msg = mp.verify_model_signature("unsig-model")
        assert not ok
        assert "No signature" in msg

    def test_verify_model_not_registered(self):
        mp = ModelProvenance()
        ok, msg = mp.verify_model_signature("nonexistent")
        assert not ok

    def test_register_checkpoint(self):
        mp = ModelProvenance()
        ckpt_hash = mp.register_checkpoint("gpt2", 100, "abc123hash", loss=2.5)
        assert ckpt_hash

    def test_get_status(self):
        mp = ModelProvenance()
        mp.register_model("test-model", "local", "alice")
        status = mp.get_status()
        assert status["models_registered"] == 1

    def test_get_provenance(self):
        mp = ModelProvenance()
        mp.register_model("prov-model", "huggingface", "eve")
        prov = mp.get_provenance("prov-model")
        assert prov is not None
        assert prov["source"] == "huggingface"

    def test_get_provenance_nonexistent(self):
        mp = ModelProvenance()
        assert mp.get_provenance("nonexistent") is None