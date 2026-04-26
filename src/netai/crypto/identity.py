"""Cryptography utilities - node identity, group keys, secure messaging."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import base64
from dataclasses import dataclass, field
from typing import Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


@dataclass
class NodeIdentity:
    node_id: str
    signing_key: ed25519.Ed25519PrivateKey = field(default=None)
    verification_key: bytes = field(default=None)
    dh_private_key: x25519.X25519PrivateKey = field(default=None)
    dh_public_key: bytes = field(default=None)
    created_at: float = 0.0

    @classmethod
    def generate(cls, node_id: str | None = None) -> "NodeIdentity":
        sk = ed25519.Ed25519PrivateKey.generate()
        vk = sk.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        dh_priv = x25519.X25519PrivateKey.generate()
        dh_pub = dh_priv.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        ident = cls(
            node_id=node_id or secrets.token_hex(8),
            signing_key=sk,
            verification_key=vk,
            dh_private_key=dh_priv,
            dh_public_key=dh_pub,
            created_at=__import__("time").time(),
        )
        return ident

    def sign(self, data: bytes) -> bytes:
        return self.signing_key.sign(data)

    def verify(self, data: bytes, signature: bytes) -> bool:
        try:
            vk = ed25519.Ed25519PublicKey.from_public_bytes(self.verification_key)
            vk.verify(signature, data)
            return True
        except Exception:
            return False

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "verification_key": base64.b64encode(self.verification_key).decode(),
            "dh_public_key": base64.b64encode(self.dh_public_key).decode(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict, private_key_data: dict | None = None) -> "NodeIdentity":
        ident = cls(
            node_id=d["node_id"],
            verification_key=base64.b64decode(d["verification_key"]),
            dh_public_key=base64.b64decode(d["dh_public_key"]),
            created_at=d.get("created_at", 0),
        )
        if private_key_data:
            ident.signing_key = ed25519.Ed25519PrivateKey.from_private_bytes(
                base64.b64decode(private_key_data["signing_key"])
            )
            ident.dh_private_key = x25519.X25519PrivateKey.from_private_bytes(
                base64.b64decode(private_key_data["dh_private_key"])
            )
        return ident

    def export_private(self) -> dict:
        return {
            "signing_key": base64.b64encode(
                self.signing_key.private_bytes(
                    serialization.Encoding.Raw,
                    serialization.PrivateFormat.Raw,
                    serialization.NoEncryption(),
                )
            ).decode(),
            "dh_private_key": base64.b64encode(
                self.dh_private_key.private_bytes(
                    serialization.Encoding.Raw,
                    serialization.PrivateFormat.Raw,
                    serialization.NoEncryption(),
                )
            ).decode(),
        }


@dataclass
class GroupKey:
    group_id: str
    shared_secret: bytes = field(default=None)
    key_version: int = 1
    created_at: float = 0.0
    members: dict[str, bytes] = field(default_factory=dict)

    def encrypt(self, plaintext: bytes, aad: bytes | None = None) -> bytes:
        if not self.shared_secret:
            raise ValueError("No shared secret set")
        nonce = os.urandom(12)
        aead = ChaCha20Poly1305(self.shared_secret)
        ciphertext = aead.encrypt(nonce, plaintext, aad)
        return nonce + ciphertext

    def decrypt(self, data: bytes, aad: bytes | None = None) -> bytes:
        if not self.shared_secret:
            raise ValueError("No shared secret set")
        nonce = data[:12]
        ciphertext = data[12:]
        aead = ChaCha20Poly1305(self.shared_secret)
        return aead.decrypt(nonce, ciphertext, aad)

    def rotate(self) -> "GroupKey":
        if not self.shared_secret:
            raise ValueError("No shared secret set")
        new_secret = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=os.urandom(16),
            info=f"group:{self.group_id}:v{self.key_version + 1}".encode(),
        ).derive(self.shared_secret + os.urandom(32))
        return GroupKey(
            group_id=self.group_id,
            shared_secret=new_secret,
            key_version=self.key_version + 1,
            created_at=__import__("time").time(),
            members=self.members.copy(),
        )


def derive_group_key(group_id: str, passphrase: str | None = None) -> GroupKey:
    if passphrase:
        secret = PBKDF2_HMAC(
            "sha256",
            passphrase.encode(),
            salt=group_id.encode(),
            iterations=600000,
            dklen=32,
        )
    else:
        secret = os.urandom(32)
    return GroupKey(
        group_id=group_id,
        shared_secret=secret,
        created_at=__import__("time").time(),
    )


def PBKDF2_HMAC(hash_name, password, salt, iterations, dklen):
    import hashlib as hl
    return hl.pbkdf2_hmac(hash_name, password, salt, iterations, dklen)


def compute_shared_secret(my_dh_private: x25519.X25519PrivateKey, their_dh_public: bytes) -> bytes:
    their_key = x25519.X25519PublicKey.from_public_bytes(their_dh_public)
    shared = my_dh_private.exchange(their_key)
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"netai-dh",
    ).derive(shared)


def hmac_verify(data: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(secret.encode(), data, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 600000, 32)
    return base64.b64encode(h).decode(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 600000, 32)
    return hmac.compare_digest(base64.b64encode(h).decode(), stored_hash)