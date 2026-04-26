"""Gradient integrity verification and Byzantine-resistant aggregation."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from typing import Any

import numpy as np

from netai.crypto.identity import NodeIdentity

logger = logging.getLogger(__name__)


class GradientIntegrityChecker:
    def __init__(self):
        self._hash_registry: dict[str, dict[int, dict[str, str]]] = {}
        self._norm_history: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._node_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "submissions": 0, "rejected": 0, "avg_norm": 0.0, "last_step": -1
        })
        self.max_gradient_norm: float = 1000.0
        self.norm_std_multiplier: float = 3.0
        self.min_nodes_for_byzantine: int = 3

    def compute_gradient_hash(self, gradient: np.ndarray, layer_name: str,
                              job_id: str, step: int, node_id: str) -> str:
        data = {
            "shape": list(gradient.shape),
            "norm": float(np.linalg.norm(gradient)),
            "mean": float(gradient.mean()),
            "std": float(gradient.std()),
            "first": float(gradient.flat[0]) if gradient.size > 0 else 0.0,
            "last": float(gradient.flat[-1]) if gradient.size > 0 else 0.0,
            "layer": layer_name,
            "job": job_id,
            "step": step,
            "node": node_id,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:32]

    def register_hash(self, job_id: str, step: int, layer_name: str, hash_value: str):
        if job_id not in self._hash_registry:
            self._hash_registry[job_id] = {}
        if step not in self._hash_registry[job_id]:
            self._hash_registry[job_id][step] = {}
        self._hash_registry[job_id][step][layer_name] = hash_value

    def verify_gradient(self, gradient: np.ndarray, layer_name: str,
                        job_id: str, step: int, node_id: str,
                        claimed_hash: str = "") -> tuple[bool, str]:
        norm = float(np.linalg.norm(gradient))
        if np.isnan(norm) or np.isinf(norm):
            return False, f"Gradient has invalid norm: {norm}"
        if norm > self.max_gradient_norm:
            return False, f"Gradient norm {norm:.1f} exceeds max {self.max_gradient_norm}"

        history = self._norm_history[job_id][layer_name]
        if len(history) >= 5:
            recent = history[-20:]
            mean_norm = float(np.mean(recent))
            std_norm = float(np.std(recent))
            if std_norm > 0 and abs(norm - mean_norm) > self.norm_std_multiplier * std_norm:
                return False, f"Gradient norm {norm:.1f} is anomalous (mean={mean_norm:.1f}, std={std_norm:.1f})"

        history.append(norm)
        if len(history) > 100:
            self._norm_history[job_id][layer_name] = history[-100:]

        actual_hash = self.compute_gradient_hash(gradient, layer_name, job_id, step, node_id)
        if claimed_hash and claimed_hash != actual_hash:
            return False, f"Hash mismatch: claimed={claimed_hash[:16]}, actual={actual_hash[:16]}"

        self.register_hash(job_id, step, layer_name, actual_hash)
        return True, "ok"

    def verify_node_gradient(self, gradient: np.ndarray, layer_name: str,
                             job_id: str, step: int, node_id: str,
                             signature: bytes | None = None,
                             signer: NodeIdentity | None = None) -> tuple[bool, str]:
        ok, msg = self.verify_gradient(gradient, layer_name, job_id, step, node_id)
        if not ok:
            return ok, msg
        if signature and signer:
            data = self.compute_gradient_hash(gradient, layer_name, job_id, step, node_id)
            if not signer.verify(data.encode(), signature):
                return False, "Gradient signature verification failed"
        return True, "ok"

    def byzantine_aggregate(self, gradients_per_node: dict[str, dict[str, np.ndarray]],
                            job_id: str = "", step: int = 0) -> dict[str, np.ndarray]:
        if not gradients_per_node:
            return {}

        node_count = len(gradients_per_node)
        layer_names = set()
        for layers in gradients_per_node.values():
            layer_names.update(layers.keys())

        verified_gradients: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
        for node_id, layers in gradients_per_node.items():
            for layer_name, grad in layers.items():
                ok, _ = self.verify_gradient(grad, layer_name, job_id, step, node_id)
                if ok:
                    verified_gradients[layer_name].append((node_id, grad))
                    self._node_stats[node_id]["submissions"] += 1
                    self._node_stats[node_id]["last_step"] = step
                else:
                    self._node_stats[node_id]["rejected"] += 1
                    logger.warning("Byzantine: rejected gradient from %s layer=%s", node_id, layer_name)

        result: dict[str, np.ndarray] = {}
        for layer_name, entries in verified_gradients.items():
            if not entries:
                continue
            arrays = [e[1] for e in entries]
            if node_count >= self.min_nodes_for_byzantine:
                norms = [float(np.linalg.norm(a)) for a in arrays]
                median_norm = float(np.median(norms))
                mad = float(np.median([abs(n - median_norm) for n in norms]))
                filtered = []
                for node_id, arr in entries:
                    n = float(np.linalg.norm(arr))
                    if mad > 0 and abs(n - median_norm) / max(mad, 1e-8) < 5.0:
                        filtered.append(arr)
                    elif mad == 0:
                        filtered.append(arr)
                if not filtered:
                    filtered = arrays
                stacked = np.stack(filtered)
                trimmed_count = max(1, len(filtered) // 5)
                if len(filtered) > 2 and trimmed_count > 0:
                    sorted_by_norm = sorted(filtered, key=lambda a: float(np.linalg.norm(a)))
                    trimmed = sorted_by_norm[trimmed_count:-trimmed_count] if trimmed_count < len(sorted_by_norm) // 2 else sorted_by_norm
                else:
                    trimmed = filtered
                result[layer_name] = np.mean(np.stack(trimmed), axis=0) if trimmed else arrays[0]
            else:
                result[layer_name] = np.mean(np.stack(arrays), axis=0)

        for node_id, stats in self._node_stats.items():
            if stats["submissions"] > 0:
                stats["avg_norm"] = sum(
                    float(np.linalg.norm(grad))
                    for grad in gradients_per_node.get(node_id, {}).values()
                ) / max(len(gradients_per_node.get(node_id, {})), 1)

        return result

    def get_node_trust_scores(self) -> dict[str, dict[str, Any]]:
        scores = {}
        for node_id, stats in self._node_stats.items():
            total = stats["submissions"] + stats["rejected"]
            reject_rate = stats["rejected"] / max(total, 1)
            trust = max(0.0, 1.0 - reject_rate * 2.0)
            scores[node_id] = {
                "submissions": stats["submissions"],
                "rejected": stats["rejected"],
                "reject_rate": round(reject_rate, 4),
                "trust_score": round(trust, 4),
                "avg_norm": round(stats["avg_norm"], 4),
            }
        return scores

    def get_status(self) -> dict[str, Any]:
        return {
            "hash_registry_jobs": len(self._hash_registry),
            "max_gradient_norm": self.max_gradient_norm,
            "norm_std_multiplier": self.norm_std_multiplier,
            "min_nodes_for_byzantine": self.min_nodes_for_byzantine,
            "node_trust_scores": self.get_node_trust_scores(),
        }


class ModelProvenance:
    def __init__(self):
        self._provenance: dict[str, dict[str, Any]] = {}
        self._checkpoint_hashes: dict[str, dict[int, str]] = defaultdict(dict)

    def register_model(self, model_id: str, source: str, owner_id: str,
                       node_identity: NodeIdentity | None = None,
                       metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        provenance = {
            "model_id": model_id,
            "source": source,
            "owner_id": owner_id,
            "registered_at": time.time(),
            "metadata": metadata or {},
        }
        if node_identity:
            provenance["signer_node_id"] = node_identity.node_id
            provenance["verification_key"] = node_identity.verification_key.hex()
            sign_payload = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
            signature = node_identity.sign(sign_payload)
            provenance["signature"] = signature.hex()

        provenance["provenance_hash"] = hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode()
        ).hexdigest()[:32]

        self._provenance[model_id] = provenance
        logger.info("Model provenance registered: %s owner=%s", model_id, owner_id)
        return provenance

    def register_checkpoint(self, model_id: str, step: int,
                            weights_hash: str, loss: float = 0.0,
                            node_identity: NodeIdentity | None = None) -> str:
        ckpt_record = {
            "model_id": model_id,
            "step": step,
            "weights_hash": weights_hash,
            "loss": loss,
            "timestamp": time.time(),
        }
        if node_identity:
            payload = json.dumps(ckpt_record, sort_keys=True, separators=(",", ":")).encode()
            sig = node_identity.sign(payload)
            ckpt_record["signature"] = sig.hex()
            ckpt_record["signer_node_id"] = node_identity.node_id

        ckpt_hash = hashlib.sha256(
            json.dumps(ckpt_record, sort_keys=True).encode()
        ).hexdigest()[:32]
        self._checkpoint_hashes[model_id][step] = ckpt_hash
        return ckpt_hash

    def verify_checkpoint(self, model_id: str, step: int,
                          weights: dict[str, np.ndarray]) -> tuple[bool, str]:
        if model_id not in self._checkpoint_hashes:
            return False, "No provenance record for this model"
        if step not in self._checkpoint_hashes[model_id]:
            return False, f"No checkpoint hash for step {step}"

        w_hash = hashlib.sha256(
            json.dumps({k: hashlib.sha256(v.tobytes()).hexdigest()
                        for k, v in sorted(weights.items())},
                       sort_keys=True).encode()
        ).hexdigest()[:32]
        expected = self._checkpoint_hashes[model_id][step]
        if w_hash != expected:
            return False, f"Checkpoint hash mismatch: expected={expected[:16]}, got={w_hash[:16]}"
        return True, "Verified"

    def verify_model_signature(self, model_id: str) -> tuple[bool, str]:
        prov = self._provenance.get(model_id)
        if not prov:
            return False, "No provenance record"
        sig_hex = prov.get("signature", "")
        if not sig_hex:
            return False, "No signature on provenance record"
        try:
            from cryptography.hazmat.primitives.asymmetric import ed25519
            vk_bytes = bytes.fromhex(prov["verification_key"])
            vk = ed25519.Ed25519PublicKey.from_public_bytes(vk_bytes)
            clean = {k: v for k, v in prov.items() if k not in ("signature", "provenance_hash")}
            payload = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
            vk.verify(bytes.fromhex(sig_hex), payload)
            return True, "Signature valid"
        except Exception as e:
            return False, f"Signature verification failed: {e}"

    def get_provenance(self, model_id: str) -> dict[str, Any] | None:
        return self._provenance.get(model_id)

    def get_status(self) -> dict[str, Any]:
        return {
            "models_registered": len(self._provenance),
            "models": {mid: {"source": p["source"], "owner": p["owner_id"],
                             "signed": "signature" in p}
                       for mid, p in self._provenance.items()},
            "checkpoints_tracked": sum(len(v) for v in self._checkpoint_hashes.values()),
        }