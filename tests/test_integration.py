"""End-to-end integration tests for the full netai stack.

Exercises: register -> login -> token -> authenticated requests for
training, inference, voting, groups, scheduling, gradient sync, security.
"""

import asyncio
import pytest
from fastapi.testclient import TestClient

from netai.api.app import create_app
from netai.p2p.network import P2PNode
from netai.training.voting import VotingEngine
from netai.training.groups import GroupManager
from netai.scheduler.scheduler import JobScheduler
from netai.security.auth import SecurityMiddleware, UserRole, Scope


def _make_app():
    sec = SecurityMiddleware()
    sec.register_user("e2e-admin", "testpassword123", UserRole.ADMIN,
                      scopes=[s.value for s in Scope])
    for ep in ["/api/vote/propose-model", "/api/vote/cast",
               "/api/pledge", "/api/group/create", "/api/group/join",
               "/api/scheduler/submit",
               "/api/inference/node/register", "/api/inference/stream",
               "/api/inference/stream-sse"]:
        sec.register_public_endpoint(ep)
    app = create_app(
        p2p_node=P2PNode(port=0, node_id="test-node"),
        voting_engine=VotingEngine(),
        group_manager=GroupManager(),
        scheduler=JobScheduler(),
        security=sec,
    )
    return app, sec


def _register_admin(client):
    resp = client.post("/api/auth/login", json={
        "user_id": "e2e-admin",
        "password": "testpassword123",
    })
    return resp.json()["access_token"]
    return resp.json()["access_token"]


def _auth_header(token):
    return {"Authorization": f"Bearer {token}"}


class TestE2EAuthFlow:
    """Test full auth flow: register, login, token creation, verification."""

    def test_register_login_verify(self):
        app, sec = _make_app()
        client = TestClient(app)
        token = _register_admin(client)
        headers = _auth_header(token)

        resp = client.get(f"/api/auth/verify?token={token}")
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

        resp = client.get("/api/auth/users", headers=headers)
        assert resp.status_code == 200
        assert "users" in resp.json()

    def test_register_duplicate_user(self):
        app, sec = _make_app()
        client = TestClient(app)
        payload = {"user_id": "dup-user", "password": "password123", "role": "user"}
        resp1 = client.post("/api/auth/register", json=payload)
        assert resp1.status_code == 200
        resp2 = client.post("/api/auth/register", json=payload)
        assert resp2.status_code == 400

    def test_login_wrong_password(self):
        app, sec = _make_app()
        client = TestClient(app)
        client.post("/api/auth/register", json={"user_id": "wp-user", "password": "password123", "role": "user"})
        resp = client.post("/api/auth/login", json={"user_id": "wp-user", "password": "wrongpassword123"})
        assert resp.status_code == 401

    def test_api_key_auth(self):
        app, sec = _make_app()
        client = TestClient(app)
        token = _register_admin(client)
        headers = _auth_header(token)
        resp = client.post("/api/auth/api-key", json={
            "user_id": "e2e-admin",
            "name": "test-key",
            "scopes": ["read", "write", "train", "admin"],
        }, headers=headers)
        assert resp.status_code == 200
        assert "api_key" in resp.json()
        api_key = resp.json()["api_key"]
        key_headers = {"X-API-Key": api_key}
        resp = client.get("/api/auth/users", headers=key_headers)
        assert resp.status_code == 200

    def test_scoped_token_access(self):
        app, sec = _make_app()
        sec.register_user("scoped-user", "password123", UserRole.OPERATOR,
                          scopes=["read", "write", "train", "inference", "vote", "gradient"])
        client = TestClient(app)
        resp = client.post("/api/auth/login", json={"user_id": "scoped-user", "password": "password123"})
        token = resp.json()["access_token"]
        headers = _auth_header(token)
        resp = client.post("/api/auth/token", json={
            "user_id": "scoped-user",
            "scopes": ["train"],
            "ttl_hours": 1.0,
        }, headers=headers)
        assert resp.status_code == 200


class TestE2ETraining:
    """Test authenticated training endpoints."""

    def test_submit_and_list_jobs(self):
        app, sec = _make_app()
        client = TestClient(app)
        token = _register_admin(client)
        headers = _auth_header(token)
        resp = client.post("/api/training/submit", json={
            "model_name": "test-model",
            "total_steps": 100,
            "hidden_size": 64,
            "num_layers": 2,
            "num_heads": 4,
        }, headers=headers)
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        assert job_id

        resp = client.get("/api/training/jobs", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert any(j["job_id"] == job_id for j in data["jobs"])


class TestE2EVoting:
    """Test voting endpoints end-to-end."""

    def test_propose_and_vote(self):
        app, sec = _make_app()
        client = TestClient(app)
        resp = client.post("/api/vote/propose-model", json={
            "model_name": "vote-test-model",
            "description": "A test model",
            "tags": ["test"],
        })
        assert resp.status_code == 200
        proposal_id = resp.json()["proposal_id"]

        resp = client.post("/api/vote/cast", json={
            "proposal_id": proposal_id,
            "voter_id": "voter1",
            "choice": "for",
            "weight": 3.0,
        })
        assert resp.status_code == 200

        resp = client.get("/api/vote/proposals")
        assert resp.status_code == 200

    def test_leaderboard_and_pledge(self):
        app, sec = _make_app()
        client = TestClient(app)
        resp = client.post("/api/pledge", json={
            "user_id": "pledger1",
            "gpu_hours": 10.0,
            "cpu_cores": 4,
            "ram_gb": 16.0,
        })
        assert resp.status_code == 200
        resp = client.get("/api/pledge/leaderboard")
        assert resp.status_code == 200


class TestE2EGroups:
    """Test group endpoints end-to-end."""

    def test_create_and_join_group(self):
        app, sec = _make_app()
        client = TestClient(app)
        resp = client.post("/api/group/create", json={
            "name": "test-group",
            "owner_id": "owner1",
            "visibility": "public",
        })
        assert resp.status_code == 200
        group_id = resp.json()["group_id"]

        resp = client.post("/api/group/join", json={
            "group_id": group_id,
            "user_id": "member1",
        })
        assert resp.status_code == 200

        resp = client.get(f"/api/group/{group_id}")
        assert resp.status_code == 200
        assert resp.json()["group_id"] == group_id

    def test_list_groups(self):
        app, sec = _make_app()
        client = TestClient(app)
        client.post("/api/group/create", json={"name": "grp1", "owner_id": "o1", "visibility": "public"})
        resp = client.get("/api/groups")
        assert resp.status_code == 200
        data = resp.json()
        assert "groups" in data
        assert len(data["groups"]) >= 1


class TestE2EInference:
    """Test inference endpoints end-to-end."""

    def test_load_run_unload(self):
        app, sec = _make_app()
        client = TestClient(app)
        token = _register_admin(client)
        headers = _auth_header(token)

        resp = client.post("/api/inference/load", json={
            "model_id": "inf-test",
            "model_name": "inf-test",
            "num_replicas": 1,
        }, headers=headers)
        assert resp.status_code == 200

        resp = client.post("/api/inference/run", json={
            "model_id": "inf-test",
            "prompt": "hello world",
            "max_tokens": 10,
        }, headers=headers)
        assert resp.status_code == 200
        result = resp.json()
        assert "text" in result

        resp = client.get("/api/inference/models", headers=headers)
        assert resp.status_code == 200


class TestE2EJackIn:
    """Test jack-in endpoint (resource registration)."""

    def test_jackin_training(self):
        app, sec = _make_app()
        client = TestClient(app)
        token = _register_admin(client)
        headers = _auth_header(token)
        resp = client.post("/api/jack-in", json={
            "user_id": "jack-user-1",
            "node_id": "jack-node-1",
            "mode": "training",
        }, headers=headers)
        assert resp.status_code == 200
        assert "training" in resp.json()["modes"]

    def test_jackin_inference(self):
        app, sec = _make_app()
        client = TestClient(app)
        token = _register_admin(client)
        headers = _auth_header(token)
        resp = client.post("/api/jack-in", json={
            "user_id": "jack-user-2",
            "node_id": "jack-node-2",
            "mode": "inference",
        }, headers=headers)
        assert resp.status_code == 200
        assert "inference" in resp.json()["modes"]

    def test_jackin_both(self):
        app, sec = _make_app()
        client = TestClient(app)
        token = _register_admin(client)
        headers = _auth_header(token)
        resp = client.post("/api/jack-in", json={
            "user_id": "jack-user-3",
            "node_id": "jack-node-3",
            "mode": "both",
        }, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "training" in data["modes"]
        assert "inference" in data["modes"]


class TestE2EGradientSync:
    """Test gradient sync endpoints."""

    def test_gradient_push(self):
        app, sec = _make_app()
        client = TestClient(app)
        token = _register_admin(client)
        headers = _auth_header(token)
        resp = client.post("/api/training/gradient-sync", json={
            "job_id": "grad-test",
            "step": 1,
            "gradients": {"layer1": [1.0, 2.0, 3.0]},
            "gradient_hash": "",
        }, headers=headers)
        assert resp.status_code == 200


class TestE2ESecurity:
    """Test security endpoints with auth."""

    def test_security_status_requires_admin(self):
        app, sec = _make_app()
        client = TestClient(app)
        token = _register_admin(client)
        headers = _auth_header(token)
        resp = client.get("/api/security/status", headers=headers)
        assert resp.status_code == 200

    def test_unauthorized_access_blocked(self):
        app, sec = _make_app()
        client = TestClient(app)
        resp = client.get("/api/security/status")
        assert resp.status_code == 401

    def test_register_short_password(self):
        app, sec = _make_app()
        client = TestClient(app)
        resp = client.post("/api/auth/register", json={
            "user_id": "shortpw",
            "password": "abc",
            "role": "user",
        })
        assert resp.status_code == 400

    def test_audit_log_accessible(self):
        app, sec = _make_app()
        client = TestClient(app)
        token = _register_admin(client)
        headers = _auth_header(token)
        resp = client.get("/api/security/audit", headers=headers)
        assert resp.status_code == 200

    def test_alerts_accessible(self):
        app, sec = _make_app()
        client = TestClient(app)
        token = _register_admin(client)
        headers = _auth_header(token)
        resp = client.get("/api/security/alerts", headers=headers)
        assert resp.status_code == 200


class TestE2ERoleScopes:
    """Test that different roles get appropriate scopes."""

    def test_operator_scopes(self):
        app, sec = _make_app()
        sec.register_user("op-user", "password123", UserRole.OPERATOR,
                          scopes=[s.value for s in Scope if s.value not in ("admin",)])
        client = TestClient(app)
        resp = client.post("/api/auth/login", json={"user_id": "op-user", "password": "password123"})
        data = resp.json()
        assert "train" in data["scopes"]
        assert "inference" in data["scopes"]
        assert "vote" in data["scopes"]

    def test_user_limited_scopes(self):
        app, sec = _make_app()
        client = TestClient(app)
        client.post("/api/auth/register", json={
            "user_id": "basic-user",
            "password": "password123",
            "role": "user",
        })
        resp = client.post("/api/auth/login", json={"user_id": "basic-user", "password": "password123"})
        data = resp.json()
        assert "read" in data["scopes"]
        assert "write" in data["scopes"]
        assert "admin" not in data["scopes"]

    def test_node_scopes(self):
        app, sec = _make_app()
        sec.register_user("node-user", "password123", UserRole.NODE,
                          scopes=["read", "write", "train", "inference", "gradient"])
        client = TestClient(app)
        resp = client.post("/api/auth/login", json={"user_id": "node-user", "password": "password123"})
        data = resp.json()
        assert "train" in data["scopes"]
        assert "inference" in data["scopes"]
        assert "gradient" in data["scopes"]


class TestE2EPublicEndpoints:
    """Test that public endpoints work without auth."""

    def test_status(self):
        app, sec = _make_app()
        client = TestClient(app)
        resp = client.get("/api/status")
        assert resp.status_code == 200

    def test_dashboard(self):
        app, sec = _make_app()
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200

    def test_resources(self):
        app, sec = _make_app()
        client = TestClient(app)
        resp = client.get("/api/resources")
        assert resp.status_code == 200

    def test_demo(self):
        app, sec = _make_app()
        client = TestClient(app)
        resp = client.get("/api/demo")
        assert resp.status_code == 200

    def test_peers(self):
        app, sec = _make_app()
        client = TestClient(app)
        resp = client.get("/api/peers")
        assert resp.status_code == 200

    def test_metrics(self):
        app, sec = _make_app()
        client = TestClient(app)
        resp = client.get("/api/metrics")
        assert resp.status_code == 200


class TestE2EResourceProfile:
    """Test resource profiling through API."""

    def test_cluster_resources(self):
        app, sec = _make_app()
        client = TestClient(app)
        resp = client.get("/api/resources/cluster")
        assert resp.status_code == 200


class TestE2EFullWorkflow:
    """Test a complete end-to-end workflow: register -> jack-in -> train -> vote -> inference."""

    def test_full_workflow(self):
        app, sec = _make_app()
        client = TestClient(app)

        # 1. Register and login as admin (created directly via security module)
        sec.register_user("workflow-user", "testpassword123", UserRole.ADMIN,
                          scopes=[s.value for s in Scope])
        resp = client.post("/api/auth/login", json={"user_id": "workflow-user", "password": "testpassword123"})
        assert resp.status_code == 200
        token = resp.json()["access_token"]
        headers = _auth_header(token)

        # 2. Check status
        resp = client.get("/api/status")
        assert resp.status_code == 200

        # 3. List resources
        resp = client.get("/api/resources")
        assert resp.status_code == 200

        # 4. Jack in for training + inference
        resp = client.post("/api/jack-in", json={
            "user_id": "workflow-user",
            "node_id": "wf-node",
            "mode": "both",
        }, headers=headers)
        assert resp.status_code == 200

        # 5. Submit training job
        resp = client.post("/api/training/submit", json={
            "model_name": "wf-model",
            "total_steps": 10,
            "hidden_size": 32,
            "num_layers": 1,
            "num_heads": 4,
        }, headers=headers)
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # 6. Check job list
        resp = client.get("/api/training/jobs", headers=headers)
        assert resp.status_code == 200
        assert any(j["job_id"] == job_id for j in resp.json()["jobs"])

        # 7. Propose a model to vote on
        resp = client.post("/api/vote/propose-model", json={
            "model_name": "wf-vote-model",
            "description": "Workflow test model",
        })
        assert resp.status_code == 200
        proposal_id = resp.json()["proposal_id"]

        # 8. Vote
        resp = client.post("/api/vote/cast", json={
            "proposal_id": proposal_id,
            "voter_id": "workflow-user",
            "choice": "for",
            "weight": 5.0,
        })
        assert resp.status_code == 200

        # 9. Create a group
        resp = client.post("/api/group/create", json={
            "name": "wf-group",
            "owner_id": "workflow-user",
            "visibility": "public",
        })
        assert resp.status_code == 200
        group_id = resp.json()["group_id"]

        # 10. Load and run inference
        resp = client.post("/api/inference/load", json={
            "model_id": "wf-infer",
            "model_name": "wf-infer",
            "num_replicas": 1,
        }, headers=headers)
        assert resp.status_code == 200

        resp = client.post("/api/inference/run", json={
            "model_id": "wf-infer",
            "prompt": "test prompt",
            "max_tokens": 5,
        }, headers=headers)
        assert resp.status_code == 200
        assert "text" in resp.json()

        # 11. Push gradient
        resp = client.post("/api/training/gradient-sync", json={
            "job_id": job_id,
            "step": 1,
            "gradients": {"layer1": [0.1, 0.2, 0.3]},
            "gradient_hash": "",
        }, headers=headers)
        assert resp.status_code == 200

        # 12. Check security
        resp = client.get("/api/security/status", headers=headers)
        assert resp.status_code == 200

        # 13. Check metrics
        resp = client.get("/api/metrics")
        assert resp.status_code == 200