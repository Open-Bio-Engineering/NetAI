"""Security framework - authentication, authorization, rate limiting, audit logging, gradient integrity."""

from netai.security.auth import (
    SecurityMiddleware,
    AuthDependency,
    AuditLogger,
    RateLimiter,
    InputValidator,
    Scope,
    UserRole,
    UserRecord,
    TokenRecord,
    ApiKeyRecord,
    require_auth,
    require_scope,
    require_admin,
    allow_public,
)
from netai.security.gradient_integrity import (
    GradientIntegrityChecker,
    ModelProvenance,
)

__all__ = [
    "SecurityMiddleware",
    "AuthDependency",
    "AuditLogger",
    "RateLimiter",
    "InputValidator",
    "Scope",
    "UserRole",
    "UserRecord",
    "TokenRecord",
    "ApiKeyRecord",
    "require_auth",
    "require_scope",
    "require_admin",
    "allow_public",
    "GradientIntegrityChecker",
    "ModelProvenance",
]