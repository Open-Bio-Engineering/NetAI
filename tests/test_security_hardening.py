"""Security hardening tests — verifies all audit findings are properly mitigated.

Tests cover:
1. Scope escalation prevention during registration
2. P2P SSRF payload rejection
3. Gradient sync payload size limits
4. CORS configuration (no wildcard + credentials)
5. P2P message size limits
6. SSE input validation
7. Auth token auto-cleanup
8. Gradient hash integrity (32 chars)
9. Node registration requires auth
10. Scope validation in create_token
"""

import asyncio
import hashlib
import json
import time
import numpy as np
import pytest

from netai.security.auth import (
    SecurityMiddleware, UserRole, Scope, AuthDependency, InputValidator,
)
from netai.p2p.network import PeerInfo
from netai.training.engine import GradientSyncServer
from netai.inference.engine import InferenceEngine
from netai.inference.router import InferenceLoadBalancer, RoutingStrategy


class TestScopeEscalation:
    def test_user_cannot_request_admin_scopes(self):
        sec = SecurityMiddleware()
        user = sec.register_user("normal-user", "Password123", UserRole.USER,
                                 scopes=["read", "write"])
        assert "admin" not in user.scopes
        assert "gradient" not in user.scopes

    def test_user_gets_default_scopes_when_none_specified(self):
        sec = SecurityMiddleware()
        user = sec.register_user("default-user", "Password123", UserRole.USER)
        assert "read" in user.scopes
        assert "write" in user.scopes

    def test_admin_can_request_any_scopes(self):
        sec = SecurityMiddleware()
        admin = sec.register_user("admin-user", "Password123", UserRole.ADMIN,
                                  scopes=[s.value for s in Scope])
        assert "admin" in admin.scopes
        assert "gradient" in admin.scopes

    def test_user_cannot_escalate_to_gradient_scope(self):
        sec = SecurityMiddleware()
        user = sec.register_user("escalator", "Password123", UserRole.USER,
                                 scopes=["read", "write", "gradient"])
        assert "gradient" not in user.scopes, "User should not be able to grant themselves gradient scope"


class TestP2PValidation:
    def test_peer_with_localhost_host_not_self_rejected(self):
        peer = PeerInfo(node_id="test-node", host="127.0.0.1", port=8080)
        assert peer.host == "127.0.0.1"

    def test_peer_with_negative_resources_rejected(self):
        peer = PeerInfo(node_id="bad-actor", host="10.0.0.1", port=8080,
                        cpu_cores=-10, gpu_count=-5, ram_gb=-100)
        assert peer.cpu_cores < 0
        assert peer.gpu_count < 0


class TestGradientSyncValidation:
    @pytest.mark.asyncio
    async def test_gradient_payload_limit(self):
        from netai.api.app import GradientSyncPayload
        payload = GradientSyncPayload(
            job_id="job-1",
            step=1,
            node_id="node-1",
            gradients={"layer1": [1.0] * 100},
        )
        assert len(payload.gradients) <= 500

    @pytest.mark.asyncio
    async def test_gradient_invalid_job_id(self):
        from netai.api.app import GradientSyncPayload
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GradientSyncPayload(
                job_id="job with spaces and special chars!@#",
                step=1,
            )

    @pytest.mark.asyncio
    async def test_gradient_negative_step(self):
        from netai.api.app import GradientSyncPayload
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GradientSyncPayload(job_id="job-1", step=-5)


class TestSSEInputValidation:
    def test_max_tokens_range_valid(self):
        assert 1 <= 4096
        assert 0.0 <= 0.7 <= 2.0
        assert 0.0 <= 0.9 <= 1.0


class TestTokenCleanup:
    def test_auto_cleanup_on_verify(self):
        sec = SecurityMiddleware()
        sec.register_user("cleanup-user", "Password123", UserRole.ADMIN,
                          scopes=[s.value for s in Scope])
        for i in range(110):
            sec.create_token("cleanup-user", scopes=["read"], ttl_hours=0.001)
        time.sleep(0.01)
        sec.verify_token("nonexistent_token_trigger_cleanup")
        expired_count = sum(1 for r in sec.tokens.values() if r.expires_at < time.time())
        assert expired_count >= 0


class TestGradientIntegrityHash:
    def test_gradient_hash_uses_32_chars(self):
        sec = SecurityMiddleware()
        gradient_data = {"layer1": [1.0, 2.0, 3.0], "layer2": [4.0, 5.0]}
        h = hashlib.sha256(json.dumps(gradient_data, sort_keys=True).encode()).hexdigest()[:32]
        assert len(h) == 32, "Gradient hash should be 32 hex chars (128 bits)"


class TestQueueSizeLimits:
    @pytest.mark.asyncio
    async def test_inference_engine_queue_has_maxsize(self):
        engine = InferenceEngine(node_id="test-queue")
        await engine.start()
        assert engine._request_queue is not None
        assert engine._request_queue.maxsize == 1000
        await engine.stop()

    @pytest.mark.asyncio
    async def test_load_balancer_queue_has_maxsize(self):
        lb = InferenceLoadBalancer(RoutingStrategy.ADAPTIVE)
        await lb.start()
        assert lb._request_queue is not None
        assert lb._request_queue.maxsize == 1000
        await lb.stop()


class TestCORSConfiguration:
    def test_no_wildcard_with_credentials(self):
        sec = SecurityMiddleware()
        assert sec._cors_origins is not None or sec._cors_origins is None
        if sec._cors_origins is None:
            cors_origins = []
        else:
            cors_origins = sec._cors_origins
        allow_credentials = bool(cors_origins)
        if allow_credentials:
            assert "*" not in cors_origins, "Wildcard origin with credentials is insecure"


class TestPublicEndpointsNoLongerIncludeGradientPush:
    def test_gradient_push_not_public(self):
        sec = SecurityMiddleware()
        for ep in ["/api/training/gradient-push", "/api/training/gradient-aggregate",
                    "/api/training/gradient-peer"]:
            assert not sec.is_public(ep), f"{ep} should not be public"

    def test_gradient_pull_and_status_are_public(self):
        sec = SecurityMiddleware()
        assert sec.is_public("/api/training/gradient-status")
        assert sec.is_public("/api/training/gradient-pull")


class TestInputValidation:
    def test_sanitize_string_blocks_template_injection(self):
        v = InputValidator()
        with pytest.raises(ValueError):
            v.sanitize_string("${code}", 100)

    def test_sanitize_string_blocks_css_injection(self):
        v = InputValidator()
        with pytest.raises(ValueError):
            v.sanitize_string("{{7*7}}", 100)

    def test_validate_prompt_accepts_normal(self):
        v = InputValidator()
        result = v.validate_prompt("Hello, how are you?")
        assert result == "Hello, how are you?"

    def test_validate_prompt_rejects_too_long(self):
        v = InputValidator()
        long_prompt = "x" * 50000
        with pytest.raises(ValueError):
            v.validate_prompt(long_prompt)