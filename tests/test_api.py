"""Tests for the web API."""

import pytest
from fastapi.testclient import TestClient
from netai.api.app import create_app
from netai.p2p.network import P2PNode
from netai.training.voting import VotingEngine
from netai.training.groups import GroupManager
from netai.scheduler.scheduler import JobScheduler
from netai.security import SecurityMiddleware, Scope, UserRole


@pytest.fixture
def client():
    sec = SecurityMiddleware()
    sec.register_user("test-user", "testpassword123", UserRole.ADMIN,
                      scopes=[s.value for s in Scope])
    sec.register_public_endpoint("/api/training/submit")
    sec.register_public_endpoint("/api/training/start")
    sec.register_public_endpoint("/api/training/stop")
    sec.register_public_endpoint("/api/inference/load")
    sec.register_public_endpoint("/api/inference/run")
    sec.register_public_endpoint("/api/inference/unload")
    sec.register_public_endpoint("/api/jack-in")
    sec.register_public_endpoint("/api/training/gradient-sync")
    sec.register_public_endpoint("/api/training/gradient-push")
    sec.register_public_endpoint("/api/training/gradient-pull")
    sec.register_public_endpoint("/api/training/gradient-aggregate")
    sec.register_public_endpoint("/api/training/gradient-peer")
    sec.register_public_endpoint("/api/vote/propose-model")
    sec.register_public_endpoint("/api/vote/cast")
    sec.register_public_endpoint("/api/pledge")
    sec.register_public_endpoint("/api/group/create")
    sec.register_public_endpoint("/api/group/join")
    sec.register_public_endpoint("/api/scheduler/submit")
    sec.register_public_endpoint("/api/inference/node/register")
    sec.register_public_endpoint("/api/inference/stream")
    sec.register_public_endpoint("/api/inference/stream-sse")
    sec.register_public_endpoint("/api/autoloader/load")
    sec.register_public_endpoint("/api/autoloader/loaded")
    sec.register_public_endpoint("/api/models/catalog")
    app = create_app(
        p2p_node=P2PNode(port=0, node_id="test-node"),
        voting_engine=VotingEngine(),
        group_manager=GroupManager(),
        scheduler=JobScheduler(),
        security=sec,
    )
    return TestClient(app)


class TestRootEndpoints:
    def test_dashboard(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "NetAI" in r.text

    def test_api_status(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200
        d = r.json()
        assert "node_id" in d
        assert "state" in d

    def test_api_resources(self, client):
        r = client.get("/api/resources")
        assert r.status_code == 200
        d = r.json()
        assert "cpu_cores" in d
        assert "ram_total_gb" in d

    def test_api_demo(self, client):
        r = client.get("/api/demo")
        assert r.status_code == 200
        d = r.json()
        assert d["name"] == "NetAI"
        assert len(d["features"]) > 0


class TestTrainingAPI:
    def test_submit_job(self, client):
        r = client.post("/api/training/submit", json={
            "model_name": "gpt2-small",
            "total_steps": 100,
            "num_layers": 2,
            "hidden_size": 32,
            "num_heads": 4,
        })
        assert r.status_code == 200
        d = r.json()
        assert "job_id" in d

    def test_list_jobs(self, client):
        client.post("/api/training/submit", json={"model_name": "test"})
        r = client.get("/api/training/jobs")
        assert r.status_code == 200
        assert len(r.json()["jobs"]) >= 1

    def test_job_status_not_found(self, client):
        r = client.get("/api/training/status/nonexistent")
        assert r.status_code == 404


class TestVotingAPI:
    def test_propose_model(self, client):
        r = client.post("/api/vote/propose-model", json={
            "model_name": "llama-7b-finetune",
            "description": "Fine-tune on domain data",
            "proposer_id": "user-1",
        })
        assert r.status_code == 200
        d = r.json()
        assert "proposal_id" in d

    def test_cast_vote(self, client):
        pr = client.post("/api/vote/propose-model", json={
            "model_name": "test-model", "proposer_id": "u1",
        }).json()
        r = client.post("/api/vote/cast", json={
            "proposal_id": pr["proposal_id"],
            "voter_id": "user-2",
            "choice": "for",
        })
        assert r.status_code == 200

    def test_list_proposals(self, client):
        r = client.get("/api/vote/proposals")
        assert r.status_code == 200
        assert "proposals" in r.json()


class TestPledgeAPI:
    def test_pledge(self, client):
        r = client.post("/api/pledge", json={
            "user_id": "user-1",
            "cpu_cores": 4,
            "gpu_count": 1,
            "ram_gb": 32,
        })
        assert r.status_code == 200
        d = r.json()
        assert "score" in d
        assert d["score"] > 0

    def test_leaderboard(self, client):
        client.post("/api/pledge", json={"user_id": "u1", "cpu_cores": 4, "gpu_count": 1, "ram_gb": 32})
        r = client.get("/api/pledge/leaderboard")
        assert r.status_code == 200
        assert len(r.json()["leaderboard"]) >= 1

    def test_cluster_resources(self, client):
        r = client.get("/api/resources/cluster")
        assert r.status_code == 200
        d = r.json()
        assert "total_cpu_cores" in d


class TestGroupAPI:
    def test_create_group(self, client):
        r = client.post("/api/group/create", json={
            "name": "test-team",
            "owner_id": "user-1",
            "visibility": "public",
        })
        assert r.status_code == 200
        d = r.json()
        assert "group_id" in d

    def test_join_group(self, client):
        cr = client.post("/api/group/create", json={
            "name": "join-test", "owner_id": "owner1",
            "visibility": "public", "require_approval": False,
        })
        gid = cr.json()["group_id"]
        r = client.post("/api/group/join", json={
            "group_id": gid, "user_id": "user-2",
            "cpu_cores": 4, "gpu_count": 0, "ram_gb": 16,
        })
        assert r.status_code == 200

    def test_list_groups(self, client):
        client.post("/api/group/create", json={"name": "g1", "owner_id": "o1"})
        r = client.get("/api/groups")
        assert r.status_code == 200
        assert len(r.json()["groups"]) >= 1

    def test_get_group(self, client):
        cr = client.post("/api/group/create", json={"name": "info-test", "owner_id": "o1"})
        gid = cr.json()["group_id"]
        r = client.get(f"/api/group/{gid}")
        assert r.status_code == 200

    def test_group_invite(self, client):
        cr = client.post("/api/group/create", json={
            "name": "inv-test", "owner_id": "owner1",
            "visibility": "private",
        })
        gid = cr.json()["group_id"]
        r = client.get(f"/api/group/{gid}/invite?inviter_id=owner1")
        assert r.status_code == 200
        assert "invite_code" in r.json()


class TestSchedulerAPI:
    def test_scheduler_submit(self, client):
        client.post("/api/pledge", json={"user_id": "s1", "node_id": "n1", "cpu_cores": 8, "ram_gb": 32})
        r = client.post("/api/scheduler/submit?name=test&min_cpu=1&min_ram=4")
        assert r.status_code == 200

    def test_scheduler_status(self, client):
        r = client.get("/api/scheduler/status")
        assert r.status_code == 200
        d = r.json()
        assert "queued" in d
        assert "nodes_registered" in d


class TestGitHubAPI:
    def test_webhook_endpoint(self, client):
        r = client.post("/api/github/webhook", json={"test": True})
        assert r.status_code in (200, 403)  # 403 if no secret configured


class TestModelsAndAutoloaderAPI:
    def test_models_catalog(self, client):
        r = client.get("/api/models/catalog")
        assert r.status_code == 200
        d = r.json()
        assert "models" in d
        assert "total_models" in d

    def test_models_catalog_size_class_filter(self, client):
        r = client.get("/api/models/catalog?size_class=mini")
        assert r.status_code == 200
        d = r.json()
        if d["models"]:
            for m in d["models"]:
                assert m["size_class"] == "mini"

    def test_models_catalog_invalid_size_class(self, client):
        r = client.get("/api/models/catalog?size_class=invalid")
        assert r.status_code == 400

    def test_models_get_specific(self, client):
        r = client.get("/api/models/catalog")
        d = r.json()
        if d["models"]:
            model_id = d["models"][0]["model_id"]
            r2 = client.get(f"/api/models/{model_id}")
            assert r2.status_code == 200
            m = r2.json()
            assert m["model_id"] == model_id
            assert "params_m" in m
            assert "vram_required_mb" in m

    def test_models_get_not_found(self, client):
        r = client.get("/api/models/nonexistent-model-xyz")
        assert r.status_code == 404

    def test_autoloader_status(self, client):
        r = client.get("/api/autoloader/status")
        assert r.status_code == 200
        d = r.json()
        assert "available_vram_mb" in d
        assert "loaded_models" in d
        assert "recommended_loads" in d
        assert "preferred_quant" in d

    def test_autoloader_load_empty(self, client):
        r = client.post("/api/autoloader/load", json={})
        assert r.status_code == 200
        d = r.json()
        assert "plan" in d
        assert "total_models" in d

    def test_autoloader_recommend(self, client):
        r = client.get("/api/autoloader/recommend?vram_mb=8000")
        assert r.status_code == 200
        d = r.json()
        assert "available_vram_mb" in d
        assert "recommended_models" in d