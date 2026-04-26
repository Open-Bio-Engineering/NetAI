"""Authentication, authorization, rate limiting, audit logging, input validation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi import Depends, HTTPException, Query, Request, Response
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from netai.crypto.identity import NodeIdentity, hmac_verify, hash_password, verify_password

logger = logging.getLogger(__name__)


class Scope(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    TRAIN = "train"
    INFERENCE = "inference"
    VOTE = "vote"
    GROUP = "group"
    GRADIENT = "gradient"


class AuthLevel(str, Enum):
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    SCOPED = "scoped"
    ADMIN = "admin"


class UserRole(str, Enum):
    ANONYMOUS = "anonymous"
    USER = "user"
    OPERATOR = "operator"
    ADMIN = "admin"
    NODE = "node"


class UserRecord(BaseModel):
    user_id: str
    role: UserRole = UserRole.USER
    password_hash: str = ""
    password_salt: str = ""
    created_at: float = Field(default_factory=time.time)
    last_login: float = 0.0
    reputation: float = 1.0
    scopes: list[str] = Field(default_factory=lambda: ["read"])
    disabled: bool = False


class TokenRecord(BaseModel):
    token: str
    user_id: str
    scopes: list[str] = Field(default_factory=lambda: ["read"])
    expires_at: float = 0.0
    created_at: float = Field(default_factory=time.time)
    refresh_token: str = ""
    node_id: str = ""


class ApiKeyRecord(BaseModel):
    key: str
    user_id: str
    name: str = "default"
    scopes: list[str] = Field(default_factory=lambda: ["read", "write"])
    created_at: float = Field(default_factory=time.time)
    last_used: float = 0.0
    disabled: bool = False


@dataclass
class RateLimitRule:
    max_requests: int = 60
    window_seconds: float = 60.0
    burst: int = 10


@dataclass
class AuditEvent:
    timestamp: float
    event_type: str
    user_id: str
    node_id: str = ""
    ip_address: str = ""
    endpoint: str = ""
    method: str = ""
    status: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0


class AuditLogger:
    def __init__(self, max_events: int = 10000):
        self._events: list[AuditEvent] = []
        self._max_events = max_events
        self._alerts: list[AuditEvent] = []
        self._failed_auth_by_ip: dict[str, list[float]] = defaultdict(list)

    def log(self, event_type: str, user_id: str = "", node_id: str = "",
            ip_address: str = "", endpoint: str = "", method: str = "",
            status: str = "", details: dict[str, Any] | None = None,
            risk_score: float = 0.0):
        event = AuditEvent(
            timestamp=time.time(),
            event_type=event_type,
            user_id=user_id,
            node_id=node_id,
            ip_address=ip_address,
            endpoint=endpoint,
            method=method,
            status=status,
            details=details or {},
            risk_score=risk_score,
        )
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        if event_type == "auth_failure":
            timestamps = self._failed_auth_by_ip[ip_address]
            timestamps.append(time.time())
            if len(timestamps) > 200:
                self._failed_auth_by_ip[ip_address] = timestamps[-100:]
        if risk_score >= 0.7 or event_type in ("auth_failure_burst", "rate_limit_exceeded",
                                                  "gradient_integrity_fail", "signature_invalid",
                                                  "tamper_detected"):
            self._alerts.append(event)
            if len(self._alerts) > 1000:
                self._alerts = self._alerts[-1000:]
            logger.warning("SECURITY ALERT: %s user=%s ip=%s endpoint=%s details=%s",
                           event_type, user_id, ip_address, endpoint, details)
        else:
            logger.info("AUDIT: %s user=%s ip=%s endpoint=%s status=%s",
                        event_type, user_id, ip_address, endpoint, status)

    def check_brute_force(self, ip_address: str, window: float = 300.0, threshold: int = 10) -> bool:
        attempts = self._failed_auth_by_ip.get(ip_address, [])
        recent = [t for t in attempts if time.time() - t < window]
        self._failed_auth_by_ip[ip_address] = recent
        return len(recent) >= threshold

    def get_recent(self, limit: int = 100, event_type: str | None = None) -> list[dict]:
        events = self._events
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        result = []
        for e in reversed(events[-limit:]):
            result.append({
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "user_id": e.user_id,
                "ip_address": e.ip_address,
                "endpoint": e.endpoint,
                "method": e.method,
                "status": e.status,
                "risk_score": e.risk_score,
                "details": e.details,
            })
        return result

    def get_alerts(self, limit: int = 50) -> list[dict]:
        return [
            {
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "user_id": e.user_id,
                "ip_address": e.ip_address,
                "endpoint": e.endpoint,
                "risk_score": e.risk_score,
                "details": e.details,
            }
            for e in reversed(self._alerts[-limit:])
        ]

    def get_stats(self) -> dict[str, Any]:
        counts: dict[str, int] = defaultdict(int)
        for e in self._events:
            counts[e.event_type] += 1
        return {
            "total_events": len(self._events),
            "total_alerts": len(self._alerts),
            "event_counts": dict(counts),
            "unique_users": len(set(e.user_id for e in self._events if e.user_id)),
            "unique_ips": len(set(e.ip_address for e in self._events if e.ip_address)),
        }


class RateLimiter:
    def __init__(self):
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._rules: dict[str, RateLimitRule] = {}
        self._blocked: dict[str, float] = {}
        self._default_rule = RateLimitRule(max_requests=60, window_seconds=60.0, burst=10)
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def set_rule(self, category: str, max_requests: int = 60,
                 window_seconds: float = 60.0, burst: int = 10):
        self._rules[category] = RateLimitRule(
            max_requests=max_requests, window_seconds=window_seconds, burst=burst
        )

    def block(self, key: str, duration: float = 3600.0):
        self._blocked[key] = time.time() + duration

    def is_blocked(self, key: str) -> bool:
        if key in self._blocked:
            if time.time() < self._blocked[key]:
                return True
            del self._blocked[key]
        return False

    async def check(self, key: str, category: str = "default") -> tuple[bool, dict[str, Any]]:
        async with self._get_lock():
            if self.is_blocked(key):
                return False, {"reason": "blocked", "retry_after": self._blocked.get(key, 0) - time.time()}
            rule = self._rules.get(category, self._default_rule)
            now = time.time()
            requests = self._requests[key]
            requests = [t for t in requests if now - t < rule.window_seconds]
            self._requests[key] = requests
            if len(requests) >= rule.max_requests:
                return False, {"reason": "rate_limited", "retry_after": rule.window_seconds}
            if len(requests) > 0 and len(requests) >= rule.burst:
                recent_burst = [t for t in requests if now - t < 1.0]
                if len(recent_burst) >= rule.burst:
                    return False, {"reason": "burst_limited", "retry_after": 1.0}
            self._requests[key].append(now)
            remaining = rule.max_requests - len(self._requests[key])
            return True, {
                "remaining": remaining,
                "limit": rule.max_requests,
                "window": rule.window_seconds,
                "reset_at": now + rule.window_seconds,
            }

    def get_status(self, key: str) -> dict[str, Any]:
        return {
            "blocked": self.is_blocked(key),
            "requests_in_window": len(self._requests.get(key, [])),
            "blocked_until": self._blocked.get(key, 0),
        }


class InputValidator:
    MAX_PROMPT_LENGTH = 32768
    MAX_MODEL_NAME_LENGTH = 256
    MAX_DESCRIPTION_LENGTH = 4096
    MAX_GRADIENT_SIZE_MB = 512
    MAX_BATCH_SIZE = 1024
    MAX_TOKENS = 8192
    VALID_DEVICE_TYPES = {"auto", "cuda", "cpu", "rocm", "mps", "vulkan"}
    VALID_ARCHITECTURES = {"transformer", "mlp", "cnn", "mamba", "rwkv", "hybrid"}
    VALID_VISIBILITIES = {"public", "private", "secret"}

    DANGEROUS_PATTERNS = [
        "\x00", "{{", "}}", "${", "<script", "javascript:",
        "data:text/html", "onerror=", "onload=", "eval(", "exec(",
    ]

    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 0) -> str:
        if not isinstance(value, str):
            raise ValueError("Expected string")
        max_len = max_length or cls.MAX_DESCRIPTION_LENGTH
        if len(value) > max_len:
            raise ValueError(f"String exceeds max length {max_len}")
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern.lower() in value.lower():
                raise ValueError(f"Input contains disallowed pattern")
        return value.strip()

    @classmethod
    def validate_prompt(cls, prompt: str) -> str:
        return cls.sanitize_string(prompt, cls.MAX_PROMPT_LENGTH)

    @classmethod
    def validate_model_name(cls, name: str) -> str:
        sanitized = cls.sanitize_string(name, cls.MAX_MODEL_NAME_LENGTH)
        if not sanitized:
            raise ValueError("Model name cannot be empty")
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./:")
        if not all(c in allowed for c in sanitized):
            raise ValueError("Model name contains invalid characters")
        return sanitized

    @classmethod
    def validate_model_id(cls, model_id: str) -> str:
        return cls.validate_model_name(model_id)

    @classmethod
    def validate_positive_int(cls, value: int, name: str, min_val: int = 1, max_val: int = 2**20) -> int:
        if not isinstance(value, int) or value < min_val or value > max_val:
            raise ValueError(f"{name} must be between {min_val} and {max_val}")
        return value

    @classmethod
    def validate_device(cls, device: str) -> str:
        if device not in cls.VALID_DEVICE_TYPES:
            raise ValueError(f"Invalid device type: {device}")
        return device

    @classmethod
    def validate_architecture(cls, arch: str) -> str:
        if arch not in cls.VALID_ARCHITECTURES:
            raise ValueError(f"Invalid architecture: {arch}")
        return arch

    @classmethod
    def validate_visibility(cls, vis: str) -> str:
        if vis not in cls.VALID_VISIBILITIES:
            raise ValueError(f"Invalid visibility: {vis}")
        return vis

    @classmethod
    def validate_user_id(cls, user_id: str) -> str:
        sanitized = cls.sanitize_string(user_id, 128)
        if not sanitized:
            raise ValueError("User ID cannot be empty")
        return sanitized

    @classmethod
    def validate_gradient_data(cls, gradients: dict, max_size_mb: float = 0) -> float:
        limit = max_size_mb or cls.MAX_GRADIENT_SIZE_MB
        total_bytes = 0
        for layer_name, data in gradients.items():
            cls.sanitize_string(layer_name, 256)
            if isinstance(data, dict):
                compressed = data.get("compressed", {})
                method = compressed.get("method", "none")
                if method not in ("topk", "quantize", "none"):
                    raise ValueError(f"Invalid compression method: {method}")
                vals = compressed.get("values", compressed.get("quantized", []))
                total_bytes += len(str(vals)) // 2
            elif hasattr(data, 'nbytes'):
                total_bytes += data.nbytes
        size_mb = total_bytes / (1024 * 1024)
        if size_mb > limit:
            raise ValueError(f"Gradient data exceeds {limit}MB limit ({size_mb:.1f}MB)")
        return size_mb


class SecurityMiddleware:
    def __init__(self, node_identity: NodeIdentity | None = None):
        self.node_identity = node_identity or NodeIdentity.generate()
        self.users: dict[str, UserRecord] = {}
        self.tokens: dict[str, TokenRecord] = {}
        self.api_keys: dict[str, ApiKeyRecord] = {}
        self.audit = AuditLogger()
        self.rate_limiter = RateLimiter()
        self.validator = InputValidator()
        self._bootstrap_secret = os.environ.get("NETAI_ADMIN_SECRET", "")
        self._bootstrap_initialized = False
        self._public_endpoints: set[str] = set()
        self._cors_origins: list[str] = []
        self._node_verifications: dict[str, bytes] = {}

        self.rate_limiter.set_rule("auth", max_requests=10, window_seconds=60.0, burst=3)
        self.rate_limiter.set_rule("inference", max_requests=120, window_seconds=60.0, burst=20)
        self.rate_limiter.set_rule("training", max_requests=20, window_seconds=60.0, burst=5)
        self.rate_limiter.set_rule("gradient", max_requests=60, window_seconds=60.0, burst=15)
        self.rate_limiter.set_rule("vote", max_requests=30, window_seconds=60.0, burst=10)
        self.rate_limiter.set_rule("default", max_requests=60, window_seconds=60.0, burst=10)

        self._init_admin()

    def _init_admin(self):
        admin_id = "admin"
        if admin_id not in self.users:
            if self._bootstrap_secret:
                pw_hash, pw_salt = hash_password(self._bootstrap_secret)
            else:
                pw_hash, pw_salt = hash_password(secrets.token_hex(16))
            self.users[admin_id] = UserRecord(
                user_id=admin_id,
                role=UserRole.ADMIN,
                password_hash=pw_hash,
                password_salt=pw_salt,
                scopes=[s.value for s in Scope],
                created_at=time.time(),
            )
            self._bootstrap_initialized = True

    ROLE_DEFAULT_SCOPES: dict[str, list[str]] = {
        "anonymous": ["read"],
        "user": ["read", "write"],
        "operator": ["read", "write", "train", "inference", "vote", "group", "gradient"],
        "admin": [s.value for s in Scope],
        "node": ["read", "write", "train", "inference", "gradient"],
    }

    def register_user(self, user_id: str, password: str, role: UserRole = UserRole.USER,
                      scopes: list[str] | None = None) -> UserRecord:
        if user_id in self.users:
            raise ValueError(f"User {user_id} already exists")
        self.validator.validate_user_id(user_id)
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")
        pw_hash, pw_salt = hash_password(password)
        default_scopes = self.ROLE_DEFAULT_SCOPES.get(role.value, [Scope.READ.value, Scope.WRITE.value])
        granted_scopes = scopes if scopes is not None else default_scopes
        granted_scopes = [s for s in granted_scopes if s in default_scopes or role == UserRole.ADMIN]
        user = UserRecord(
            user_id=user_id,
            role=role,
            password_hash=pw_hash,
            password_salt=pw_salt,
            scopes=granted_scopes,
        )
        self.users[user_id] = user
        self.audit.log("user_registered", user_id=user_id, details={"role": role.value})
        return user

    def authenticate_password(self, user_id: str, password: str) -> UserRecord | None:
        user = self.users.get(user_id)
        if not user or user.disabled:
            return None
        if not verify_password(password, user.password_hash, user.password_salt):
            return None
        user.last_login = time.time()
        return user

    def create_token(self, user_id: str, scopes: list[str] | None = None,
                     ttl_hours: float = 24.0, node_id: str = "") -> TokenRecord:
        user = self.users.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        if user.disabled:
            raise ValueError(f"User {user_id} is disabled")
        requested = scopes or user.scopes
        granted = [s for s in requested if s in user.scopes or user.role == UserRole.ADMIN]
        if user.role == UserRole.ADMIN:
            granted = requested
        token_str = secrets.token_urlsafe(32)
        refresh = secrets.token_urlsafe(32)
        record = TokenRecord(
            token=token_str,
            user_id=user_id,
            scopes=granted,
            expires_at=time.time() + ttl_hours * 3600,
            refresh_token=refresh,
            node_id=node_id,
        )
        self.tokens[token_str] = record
        self.audit.log("token_created", user_id=user_id, details={"scopes": granted, "ttl_hours": ttl_hours})
        return record

    def create_api_key(self, user_id: str, name: str = "default",
                       scopes: list[str] | None = None) -> ApiKeyRecord:
        user = self.users.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        if user.disabled:
            raise ValueError(f"User {user_id} is disabled")
        requested = scopes or [Scope.READ.value, Scope.WRITE.value]
        granted = [s for s in requested if s in user.scopes or user.role == UserRole.ADMIN]
        if user.role == UserRole.ADMIN:
            granted = requested
        key_str = f"nx_{secrets.token_urlsafe(8)}_{secrets.token_urlsafe(24)}"
        record = ApiKeyRecord(
            key=key_str,
            user_id=user_id,
            name=name,
            scopes=granted,
        )
        self.api_keys[key_str] = record
        self.audit.log("api_key_created", user_id=user_id, details={"name": name, "scopes": granted})
        return record

    def verify_token(self, token: str) -> TokenRecord | None:
        if len(self.tokens) % 100 == 0:
            self._cleanup_tokens()
        record = self.tokens.get(token)
        if not record:
            return None
        if time.time() > record.expires_at:
            del self.tokens[token]
            self.audit.log("token_expired", user_id=record.user_id, risk_score=0.1)
            return None
        user = self.users.get(record.user_id)
        if not user or user.disabled:
            del self.tokens[token]
            return None
        return record

    def _cleanup_tokens(self):
        now = time.time()
        expired = [t for t, r in self.tokens.items() if now > r.expires_at]
        for t in expired:
            del self.tokens[t]
        expired_keys = [k for k, r in self.api_keys.items() if r.expires_at and now > r.expires_at]
        for k in expired_keys:
            del self.api_keys[k]

    def cleanup_expired_tokens(self) -> int:
        now = time.time()
        expired = [t for t, r in self.tokens.items() if now > r.expires_at]
        for t in expired:
            del self.tokens[t]
        if expired:
            self.audit.log("token_cleanup", details={"count": len(expired)})
        return len(expired)

    def verify_api_key(self, key: str) -> ApiKeyRecord | None:
        record = self.api_keys.get(key)
        if not record:
            return None
        user = self.users.get(record.user_id)
        if not user or user.disabled or record.disabled:
            return None
        record.last_used = time.time()
        return record

    def revoke_token(self, token: str) -> bool:
        record = self.tokens.pop(token, None)
        if record:
            self.audit.log("token_revoked", user_id=record.user_id)
            return True
        return False

    def revoke_api_key(self, key: str) -> bool:
        record = self.api_keys.pop(key, None)
        if record:
            self.audit.log("api_key_revoked", user_id=record.user_id)
            return True
        return False

    def check_permission(self, identity: TokenRecord | ApiKeyRecord, required_scope: str) -> bool:
        user = self.users.get(identity.user_id)
        if not user:
            return False
        if user.role == UserRole.ADMIN:
            return True
        if user.role == UserRole.OPERATOR and required_scope in (Scope.READ.value, Scope.WRITE.value,
                                                                    Scope.TRAIN.value, Scope.INFERENCE.value):
            return True
        return required_scope in identity.scopes

    def register_public_endpoint(self, path: str):
        self._public_endpoints.add(path)

    def is_public(self, path: str) -> bool:
        if path in self._public_endpoints:
            return True
        for prefix in self._public_endpoints:
            if path.startswith(prefix + "/") or path.startswith(prefix + "?"):
                return True
        public_prefixes = ("/", "/api/demo", "/api/status", "/api/metrics", "/api/auth/login",
                           "/api/auth/verify", "/p2p", "/api/training/gradient-status",
                           "/api/training/gradient-pull")
        for prefix in public_prefixes:
            if path == prefix or path == prefix + "/" or path.startswith(prefix + "/") or path.startswith(prefix + "?"):
                return True
        return False

    def set_cors_origins(self, origins: list[str]):
        self._cors_origins = origins

    def sign_p2p_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload["sender_id"] = self.node_identity.node_id
        payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        signature = self.node_identity.sign(payload_bytes)
        payload["signature"] = signature.hex()
        payload["signature_algo"] = "ed25519"
        return payload

    def verify_p2p_message(self, payload: dict[str, Any], sender_public_key: bytes | None = None) -> bool:
        sig_hex = payload.get("signature", "")
        if not sig_hex:
            return False
        clean = {k: v for k, v in payload.items() if k not in ("signature", "signature_algo")}
        payload_bytes = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
        try:
            sig_bytes = bytes.fromhex(sig_hex)
            if sender_public_key:
                from cryptography.hazmat.primitives.asymmetric import ed25519 as ed
                vk = ed.Ed25519PublicKey.from_public_bytes(sender_public_key)
                vk.verify(sig_bytes, payload_bytes)
                return True
            sender_id = payload.get("sender_id", "")
            if sender_id in self._node_verifications:
                vk = ed.Ed25519PublicKey.from_public_bytes(self._node_verifications[sender_id])
                vk.verify(sig_bytes, payload_bytes)
                return True
            return False
        except Exception:
            return False

    def register_node_public_key(self, node_id: str, public_key: bytes):
        self._node_verifications[node_id] = public_key

    def verify_gradient_integrity(self, gradient_data: dict, expected_hash: str) -> bool:
        actual = hashlib.sha256(json.dumps(gradient_data, sort_keys=True).encode()).hexdigest()[:32]
        if not hmac.compare_digest(actual, expected_hash):
            self.audit.log("gradient_integrity_fail", details={"expected": expected_hash, "actual": actual},
                           risk_score=0.9)
            return False
        return True

    def get_security_status(self) -> dict[str, Any]:
        return {
            "node_id": self.node_identity.node_id,
            "users_registered": len(self.users),
            "tokens_active": len(self.tokens),
            "api_keys_active": len(self.api_keys),
            "public_endpoints": list(self._public_endpoints),
            "rate_limits": {k: {"max": r.max_requests, "window": r.window_seconds}
                            for k, r in self.rate_limiter._rules.items()},
            "audit_stats": self.audit.get_stats(),
            "node_verifications_registered": len(self._node_verifications),
        }


class AuthDependency:
    def __init__(self, security: SecurityMiddleware, required_scope: str | None = None,
                 allow_unauthenticated: bool = False):
        self.security = security
        self.required_scope = required_scope
        self.allow_unauthenticated = allow_unauthenticated

    async def __call__(self, request: Request) -> TokenRecord | ApiKeyRecord | None:
        client_ip = request.client.host if request.client else "unknown"
        endpoint = request.url.path
        method = request.method

        auth_header = request.headers.get("Authorization", "")
        api_key_header = request.headers.get("X-API-Key", "")

        identity = None

        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            identity = self.security.verify_token(token)
        elif api_key_header:
            identity = self.security.verify_api_key(api_key_header)

        if self.allow_unauthenticated and identity is None:
            return None

        if identity is None:
            if self.security.is_public(endpoint):
                return None
            auth_type = "invalid_auth" if (auth_header or api_key_header) else "no_auth"
            risk = 0.4 if auth_type == "invalid_auth" else 0.2
            self.security.audit.log("auth_failure", ip_address=client_ip,
                                    endpoint=endpoint, method=method,
                                    details={"type": auth_type},
                                    risk_score=risk)
            raise HTTPException(401, "Authentication required")

        if identity and self.required_scope:
            if not self.security.check_permission(identity, self.required_scope):
                self.security.audit.log("auth_forbidden", user_id=identity.user_id,
                                        ip_address=client_ip, endpoint=endpoint,
                                        method=method,
                                        details={"required_scope": self.required_scope,
                                                 "granted_scopes": identity.scopes},
                                        risk_score=0.5)
                raise HTTPException(403, f"Insufficient scope: {self.required_scope}")

        category = "default"
        if "/inference" in endpoint:
            category = "inference"
        elif "/training" in endpoint:
            category = "training"
        elif "/gradient" in endpoint:
            category = "gradient"
        elif "/vote" in endpoint:
            category = "vote"
        elif "/auth" in endpoint:
            category = "auth"

        ok, info = await self.security.rate_limiter.check(
            f"{identity.user_id}:{client_ip}" if identity else client_ip,
            category
        )
        if not ok:
            self.security.audit.log("rate_limit_exceeded",
                                    user_id=identity.user_id if identity else "",
                                    ip_address=client_ip, endpoint=endpoint,
                                    risk_score=0.6)
            raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": str(int(info.get("retry_after", 60)))})

        if identity:
            self.security.audit.log("auth_success", user_id=identity.user_id,
                                    ip_address=client_ip, endpoint=endpoint, method=method,
                                    status="ok")

        return identity


def require_auth(security: SecurityMiddleware) -> AuthDependency:
    return AuthDependency(security, required_scope=None, allow_unauthenticated=False)


def require_scope(security: SecurityMiddleware, scope: str) -> AuthDependency:
    return AuthDependency(security, required_scope=scope, allow_unauthenticated=False)


def require_admin(security: SecurityMiddleware) -> AuthDependency:
    return AuthDependency(security, required_scope=Scope.ADMIN.value, allow_unauthenticated=False)


def allow_public(security: SecurityMiddleware) -> AuthDependency:
    return AuthDependency(security, allow_unauthenticated=True)