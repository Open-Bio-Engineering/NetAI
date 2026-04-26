"""CLI interface for NetAI distributed AI training and inference."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import stat
import sys
import time
from typing import Any

import httpx


BASE_URL = "http://127.0.0.1:8001"
TOKEN_FILE = os.path.join(os.path.expanduser("~"), ".config", "netai", "token")

_args_url = BASE_URL


def _load_token() -> str:
    try:
        with open(TOKEN_FILE) as f:
            data = json.load(f)
            return data.get("token", "")
    except Exception:
        return ""


def _save_token(token: str, user_id: str):
    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    with open(TOKEN_FILE, "w") as f:
        json.dump({"token": token, "user_id": user_id}, f)
    try:
        os.chmod(TOKEN_FILE, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    token = _load_token()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get(path: str) -> dict:
    try:
        r = httpx.get(f"{_args_url}{path}", headers=_headers(), timeout=10)
        if r.status_code == 401:
            return {"error": "Authentication required. Run: netai auth login <user> <password>"}
        if r.status_code == 403:
            return {"error": "Forbidden. Insufficient permissions."}
        if r.status_code == 404:
            return {"error": "Not found."}
        return r.json()
    except httpx.ConnectError:
        return {"error": f"Cannot connect to {_args_url}. Is the server running?"}
    except Exception as e:
        return {"error": str(e)}


def _post(path: str, data: dict | None = None) -> dict:
    try:
        r = httpx.post(f"{_args_url}{path}", json=data or {}, headers=_headers(), timeout=30)
        if r.status_code == 401:
            return {"error": "Authentication required. Run: netai auth login <user> <password>"}
        if r.status_code == 403:
            return {"error": "Forbidden. Insufficient permissions."}
        if r.status_code == 429:
            return {"error": "Rate limited. Try again later."}
        if r.status_code >= 400:
            try:
                body = r.json()
                if "detail" in body:
                    return {"error": body["detail"]}
                return body
            except Exception:
                return {"error": f"HTTP {r.status_code}"}
        return r.json()
    except httpx.ConnectError:
        return {"error": f"Cannot connect to {_args_url}. Is the server running?"}
    except Exception as e:
        return {"error": str(e)}


def _err(d: dict):
    error_val = d.get("error")
    detail_val = d.get("detail")
    if error_val and isinstance(error_val, str) and error_val.strip():
        print(f"Error: {error_val}")
        return True
    if error_val and isinstance(error_val, dict):
        print(f"Error: {error_val}")
        return True
    if detail_val and isinstance(detail_val, str) and detail_val.strip():
        print(f"Error: {detail_val}")
        return True
    return False


def cmd_status(args):
    d = _get("/api/status")
    if _err(d):
        return
    p = d.get("profile", {})
    print(f"  Node:     {d.get('node_id', '-')}")
    print(f"  State:    {d.get('state', '-')}")
    print(f"  CPU:      {p.get('cpu_available', 0)}/{p.get('cpu_cores', 0)} cores")
    print(f"  GPU:      {p.get('gpu_count', 0)} ({', '.join(p.get('gpu_names', [])) or 'none'})")
    print(f"  RAM:      {p.get('ram_available_gb', 0):.1f}/{p.get('ram_total_gb', 0):.1f} GB")
    print(f"  CUDA:     {'yes' if p.get('has_cuda') else 'no'} | ROCm: {'yes' if p.get('has_rocm') else 'no'} | Vulkan: {'yes' if p.get('has_vulkan') else 'no'}")
    print(f"  PyTorch:  {'yes' if p.get('torch_available') else 'no'}")
    print(f"  Peers:    {d.get('peer_count', 0)}")
    print(f"  Jobs:     {len(d.get('jobs', []))}")
    print(f"  Groups:   {d.get('groups', 0)}")
    print(f"  Proposals:{d.get('proposals', 0)}")
    print(f"  Pledges:  {d.get('pledges', 0)}")


def cmd_peers(args):
    d = _get("/api/peers")
    peers = d.get("peers", [])
    if not peers:
        print("No peers connected")
        return
    print(f"{'Node ID':<18} {'Host':<22} {'State':<12} {'CPU':<10} {'GPU':<6} {'RAM':<10}")
    print("-" * 78)
    for p in peers:
        print(f"{p.get('node_id', '-'):<18} {p.get('host', '-')}:{p.get('port', 0)} "
              f"{p.get('state', '-'):<12} {p.get('cpu_avail', 0)}/{p.get('cpu_cores', 0)} "
              f"{p.get('gpu_avail', 0)}/{p.get('gpu_count', 0)} "
              f"{p.get('ram_avail_gb', 0):.1f}GB")


def cmd_train(args):
    data = {
        "model_name": args.model,
        "model_architecture": args.arch or "transformer",
        "hidden_size": args.hidden or 768,
        "num_layers": args.layers or 12,
        "total_steps": args.steps or 1000,
        "batch_size": args.batch or 8,
        "learning_rate": args.lr or 3e-4,
        "device_preference": args.device or "cuda",
        "group_id": args.group or "",
    }
    r = _post("/api/training/submit", data)
    if _err(r):
        return
    job_id = r.get("job_id", "-")
    print(f"Job submitted: {job_id}")
    if args.start:
        s = _post(f"/api/training/start/{job_id}")
        if not _err(s):
            print(f"Training started: {s.get('status', '-')}")
    if args.watch:
        _watch_job(job_id)


def _watch_job(job_id: str, interval: float = 2.0, max_loops: int = 500):
    print(f"Watching job {job_id}... (Ctrl+C to stop)")
    try:
        for _ in range(max_loops):
            d = _get(f"/api/training/status/{job_id}")
            if _err(d):
                break
            status = d.get("status", "-")
            step = d.get("step", 0)
            loss = d.get("loss", "-")
            total = d.get("config", {}).get("total_steps", "?")
            elapsed = d.get("elapsed_s", 0)
            print(f"\r  Step {step}/{total} | Loss: {loss} | Status: {status} | {elapsed:.0f}s", end="", flush=True)
            if status in ("completed", "failed", "cancelled"):
                print()
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


def cmd_jobs(args):
    d = _get("/api/training/jobs")
    jobs = d.get("jobs", [])
    if not jobs:
        print("No jobs")
        return
    print(f"{'Job ID':<14} {'Model':<20} {'Status':<12} {'Step':<8} {'Loss':<10} {'Elapsed':<10}")
    print("-" * 74)
    for j in jobs:
        cfg = j.get("config", {})
        loss = j.get("loss")
        loss_str = f"{loss:.4f}" if loss is not None else "-"
        print(f"{j.get('job_id', '-'):<14} {cfg.get('model_name', '-')[:20]:<20} "
              f"{j.get('status', '-'):<12} {j.get('step', 0):<8} "
              f"{loss_str:<10} {j.get('elapsed_s', 0):.0f}s")


def cmd_vote(args):
    if args.action == "propose":
        data = {
            "model_name": args.model or "custom",
            "architecture": args.arch or "transformer",
            "description": args.desc or "",
            "proposer_id": args.user or "anonymous",
            "group_id": args.group or "",
        }
        r = _post("/api/vote/propose-model", data)
        if _err(r):
            return
        print(f"Proposal: {r.get('proposal_id', '-')}")
    elif args.action == "cast":
        data = {
            "proposal_id": args.proposal,
            "voter_id": args.user or "anonymous",
            "choice": args.choice or "for",
        }
        r = _post("/api/vote/cast", data)
        if _err(r):
            return
        print(f"Vote cast: {r.get('vote_id', '-')} (weight: {r.get('weight', 0):.2f})")
    elif args.action == "list":
        d = _get("/api/vote/proposals")
        proposals = d.get("proposals", [])
        if not proposals:
            print("No proposals")
            return
        print(f"{'ID':<14} {'Title':<30} {'Status':<12} {'For':<8} {'Against':<8}")
        print("-" * 72)
        for p in proposals:
            print(f"{p.get('proposal_id', '-'):<14} {p.get('title', '-')[:30]:<30} "
                  f"{p.get('status', '-'):<12} {p.get('weighted_for', 0):<8.1f} "
                  f"{p.get('weighted_against', 0):<8.1f}")


def cmd_pledge(args):
    data = {
        "user_id": args.user or "anonymous",
        "node_id": args.node or "",
        "cpu_cores": args.cpu or 0,
        "gpu_count": args.gpu or 0,
        "ram_gb": args.ram or 0,
        "time_hours": args.hours or 24,
        "group_id": args.group or "",
    }
    r = _post("/api/pledge", data)
    if _err(r):
        return
    print(f"Pledged! Score: {r.get('score', 0):.1f} | {r.get('summary', '')}")


def cmd_group(args):
    if args.action == "create":
        data = {
            "name": args.name,
            "owner_id": args.owner,
            "description": args.desc or "",
            "visibility": args.visibility or "private",
            "max_members": args.max_members or 100,
            "require_approval": args.approval != "false",
            "passphrase": args.passphrase or "",
        }
        r = _post("/api/group/create", data)
        if _err(r):
            return
        print(f"Group created: {r.get('group_id', '-')} ({r.get('name', '')})")
    elif args.action == "list":
        d = _get("/api/groups")
        groups = d.get("groups", [])
        if not groups:
            print("No groups")
            return
        print(f"{'ID':<14} {'Name':<25} {'Visibility':<12} {'Members':<10}")
        print("-" * 61)
        for g in groups:
            print(f"{g.get('group_id', '-'):<14} {g.get('name', '-')[:25]:<25} "
                  f"{g.get('visibility', '-'):<12} {g.get('members', 0):<10}")
    elif args.action == "info":
        d = _get(f"/api/group/{args.group_id}")
        if _err(d):
            return
        print(f"  Group:     {d.get('group_name', '-')} ({d.get('group_id', '-')})")
        print(f"  Members:   {d.get('members', 0)}")
        print(f"  CPU:       {d.get('pledged_cpu_cores', 0)} cores")
        print(f"  GPU:       {d.get('pledged_gpu_count', 0)}")
        print(f"  RAM:       {d.get('pledged_ram_gb', 0):.0f} GB")
        print(f"  Jobs:      {d.get('active_jobs', 0)}/{d.get('max_concurrent_jobs', 0)}")
        details = d.get("member_details", [])
        if details:
            print(f"  Members:")
            for m in details:
                print(f"    {m.get('user_id', '-'):20} role={m.get('role', '-'):8} rep={m.get('reputation', 0):.1f}")
    elif args.action == "join":
        data = {
            "group_id": args.group_id,
            "user_id": args.user or "anonymous",
            "invite_code": args.invite or "",
            "cpu_cores": args.cpu or 0,
            "gpu_count": args.gpu or 0,
            "ram_gb": args.ram or 0,
        }
        r = _post("/api/group/join", data)
        if _err(r):
            return
        print(f"Joined {r.get('group_id', args.group_id)}: {r.get('message', 'ok')}")
    elif args.action == "invite":
        r = _get(f"/api/group/{args.group_id}/invite?inviter_id={args.inviter}")
        if "invite_code" in r:
            print(f"Invite code: {r['invite_code']}")
        else:
            _err(r) or print("Error: Not authorized or group not found")
    elif args.action == "propose-training":
        from urllib.parse import quote
        url = f"/api/group/{quote(args.group_id)}/propose-training?proposer_id={quote(args.proposer)}&model_name={quote(args.model or 'gpt2-small')}&steps={args.steps or 1000}"
        r = _post(url)
        if _err(r):
            return
        print(f"Training proposal: {r.get('proposal_id', '-')} ({r.get('status', '-')})")


def cmd_inference(args):
    if args.action == "load":
        model_id = args.model or args.name
        data = {
            "model_id": model_id,
            "model_name": args.name or args.model,
            "version": args.version or "latest",
            "num_shards": args.shards or 1,
            "device": args.device or "auto",
            "mirror_enabled": (args.mirror or "true") == "true",
        }
        r = _post("/api/inference/load", data)
        if _err(r):
            return
        print(f"Model loaded: {r.get('model_id', '-')} (shards: {r.get('shards', 1)})")
    elif args.action == "run":
        data = {
            "model_id": args.model,
            "prompt": args.prompt or "",
            "max_tokens": args.max_tokens or 256,
            "temperature": args.temperature or 0.7,
            "top_p": args.top_p or 0.9,
        }
        r = _post("/api/inference/run", data)
        if _err(r):
            return
        print(r.get("text", r.get("error", "No output")))
        if args.verbose:
            print(f"\n  Tokens: {r.get('tokens_generated', 0)} | Latency: {r.get('latency_ms', 0):.0f}ms | Speed: {r.get('tokens_per_second', 0):.0f} tok/s")
    elif args.action == "unload":
        model_id = args.model
        r = _post(f"/api/inference/unload/{model_id}")
        if _err(r):
            return
        unloaded = r.get("unloaded", False)
        print(f"Unloaded: {model_id}" if unloaded else f"Failed to unload: {model_id}")
    elif args.action == "status":
        d = _get("/api/inference/status")
        if _err(d):
            return
        local = d.get("local", {})
        cluster = d.get("cluster", {})
        print(f"  Local models:    {local.get('models_loaded', 0)}")
        print(f"  Total inferences:{local.get('total_inferences', 0)}")
        print(f"  Cluster nodes:   {cluster.get('total_nodes', 0)} ({cluster.get('available_nodes', 0)} available)")
        print(f"  Strategy:        {cluster.get('strategy', '-')}")
    elif args.action == "models":
        d = _get("/api/inference/models")
        if _err(d):
            return
        models = d.get("models", [])
        details = d.get("details", {})
        if not models:
            print("No models loaded")
            return
        print(f"{'Model ID':<30} {'Version':<12} {'Shards':<8}")
        print("-" * 50)
        for mid in models:
            det = details.get(mid, {})
            print(f"{mid:<30} {det.get('version', '-'):<12} {det.get('shards', '-'):<8}")


def cmd_jackin(args):
    models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    data = {
        "user_id": args.user or "anonymous",
        "mode": args.mode or "both",
        "cpu_cores": args.cpu or 0,
        "gpu_count": args.gpu or 0,
        "ram_gb": args.ram or 0,
        "time_hours": args.hours or 24,
        "group_id": args.group or "",
        "models_to_serve": models,
    }
    r = _post("/api/jack-in", data)
    if _err(r):
        return
    modes = r.get("modes", [])
    print(f"Jacked in! Modes: {' + '.join(modes)}")
    if r.get("pledge_score"):
        print(f"  Pledge score: {r['pledge_score']:.1f} ({r.get('pledge_summary', '')})")
    if r.get("group_joined"):
        print(f"  Joined group: {r['group_joined']}")
    if r.get("profile"):
        print(f"  Profile: CPU {r['profile']['cpu']} | GPU {r['profile']['gpu']} | RAM {r['profile']['ram']}")


def cmd_gradient(args):
    if args.action == "status":
        d = _get("/api/training/gradient-status")
        if _err(d):
            return
        print(f"  Node:   {d.get('node_id', '-')}")
        print(f"  Peers:  {d.get('peers', 0)}")
        print(f"  Running:{d.get('running', False)}")
        gs = d.get("gradient_store", {})
        if gs:
            for jid, info in gs.items():
                steps = info.get("steps_available", [])
                agg = info.get("aggregated_step", -1)
                print(f"  Job {jid}: steps={steps} latest_agg={agg}")
                nps = info.get("nodes_per_step", {})
                for step, nodes in nps.items():
                    print(f"    Step {step}: nodes={nodes}")
        else:
            print("  No gradient data")
    elif args.action == "push":
        r = _post(f"/api/training/gradient-push/{args.job}/{args.step}")
        if _err(r):
            return
        print(f"Push: {r.get('status', '-')} (layers: {r.get('layers', 0)})")
    elif args.action == "pull":
        d = _get(f"/api/training/gradient-pull/{args.job}/{args.step}")
        if _err(d):
            return
        grads = d.get("gradients", {})
        if grads:
            for layer, g in grads.items():
                print(f"  {layer}: shape={g.get('shape', [])} mean={g.get('mean', 0):.4f} std={g.get('std', 0):.4f} norm={g.get('norm', 0):.2f}")
        else:
            print("No gradient data")
    elif args.action == "aggregate":
        r = _post(f"/api/training/gradient-aggregate/{args.job}/{args.step}")
        if _err(r):
            return
        print(f"Aggregated: {r.get('status', '-')} (layers: {r.get('layers', 0)})")
    elif args.action == "peer":
        r = _post(f"/api/training/gradient-peer?node_id={args.peer_id}&endpoint={args.endpoint}")
        if _err(r):
            return
        print(f"Peer added: {args.peer_id} @ {args.endpoint}")
    elif args.action == "sync":
        data = {
            "job_id": args.job,
            "step": args.step,
            "node_id": args.node_id or "",
            "gradients": {},
            "gradient_hash": args.hash or "",
        }
        r = _post("/api/training/gradient-sync", data)
        if _err(r):
            return
        print(f"Sync: {r.get('status', '-')} (job: {r.get('job_id', '-')}, step: {r.get('step', '-')})")


def cmd_auth(args):
    if args.action == "register":
        data = {"user_id": args.user, "password": args.password, "role": args.role or "user"}
        if args.scopes:
            data["scopes"] = [s.strip() for s in args.scopes.split(",") if s.strip()]
        r = _post("/api/auth/register", data)
        if _err(r):
            return
        print(f"Registered: {r.get('user_id', '-')} (role: {r.get('role', '-')})")
    elif args.action == "login":
        data = {"user_id": args.user, "password": args.password}
        r = _post("/api/auth/login", data)
        if _err(r):
            return
        token = r.get("access_token", "")
        _save_token(token, args.user)
        print(f"Logged in: {r.get('user_id', '-')}")
        print(f"Token saved to {TOKEN_FILE}")
        print(f"Scopes: {r.get('scopes', [])}")
    elif args.action == "token":
        data = {"user_id": args.user, "scopes": (args.scopes or "read,write").split(","), "ttl_hours": args.ttl or 24.0}
        r = _post("/api/auth/token", data)
        if _err(r):
            return
        print(f"Token: {r.get('token', '-')}")
        print(f"Expires: {r.get('expires_in_hours', 0)}h | Scopes: {r.get('scopes', [])}")
    elif args.action == "apikey":
        data = {"user_id": args.user, "name": args.name or "default", "scopes": (args.scopes or "read,write").split(",")}
        r = _post("/api/auth/api-key", data)
        if _err(r):
            return
        print(f"API Key: {r.get('api_key', '-')}")
        print(f"Name: {r.get('name', '-')} | Scopes: {r.get('scopes', [])}")
    elif args.action == "verify":
        token = args.token or _load_token()
        if not token:
            print("No token. Run: netai auth login")
            return
        d = _get(f"/api/auth/verify?token={token}")
        if d.get("valid"):
            print(f"Valid token for: {d.get('user_id', '-')} (scopes: {d.get('scopes', [])})")
        else:
            print("Invalid or expired token")
    elif args.action == "logout":
        try:
            os.remove(TOKEN_FILE)
            print("Logged out (token removed)")
        except FileNotFoundError:
            print("No saved token")
    elif args.action == "users":
        d = _get("/api/auth/users")
        if _err(d):
            return
        users = d.get("users", [])
        if not users:
            print("No users")
            return
        print(f"{'User ID':<20} {'Role':<12} {'Scopes':<30} {'Disabled':<10}")
        print("-" * 72)
        for u in users:
            scopes = ", ".join(u.get("scopes", []))
            print(f"{u.get('user_id', '-'):<20} {u.get('role', '-'):<12} {scopes[:30]:<30} {u.get('disabled', False):<10}")


def cmd_security(args):
    if args.action == "status":
        d = _get("/api/security/status")
        if _err(d):
            return
        sec = d.get("security", {})
        gi = d.get("gradient_integrity", {})
        mp = d.get("model_provenance", {})
        print("  Security:")
        print(f"    Node:           {sec.get('node_id', '-')}")
        print(f"    Users:          {sec.get('users_registered', 0)}")
        print(f"    Active tokens:  {sec.get('tokens_active', 0)}")
        print(f"    Active API keys:{sec.get('api_keys_active', 0)}")
        print(f"    Node verifs:    {sec.get('node_verifications_registered', 0)}")
        pub = sec.get("public_endpoints", [])
        if pub:
            print(f"    Public endpoints: {', '.join(pub)}")
        audit_stats = sec.get("audit_stats", {})
        if audit_stats:
            print(f"    Audit events:   {audit_stats.get('total_events', 0)}")
        print("  Gradient Integrity:")
        print(f"    Hash registry:  {gi.get('hash_registry_jobs', 0)} jobs")
        print(f"    Max grad norm:  {gi.get('max_gradient_norm', 0)}")
        print(f"    Norm std mult:  {gi.get('norm_std_multiplier', 0)}")
        print(f"    Min Byzantine:  {gi.get('min_nodes_for_byzantine', 0)} nodes")
        scores = gi.get("node_trust_scores", {})
        if scores:
            for nid, s in scores.items():
                print(f"    {nid}: trust={s.get('trust_score', 0):.2f} subs={s.get('submissions', 0)} rejections={s.get('rejections', 0)}")
        print("  Model Provenance:")
        print(f"    Models:         {mp.get('models_registered', 0)}")
        models = mp.get("models", {})
        if models:
            for mid, info in models.items():
                print(f"      {mid}: source={info.get('source', '-')} owner={info.get('owner', '-')} signed={info.get('signed', False)}")
        print(f"    Checkpoints:    {mp.get('checkpoints_tracked', 0)}")
    elif args.action == "audit":
        limit = args.limit or 20
        d = _get(f"/api/security/audit?limit={limit}")
        if _err(d):
            return
        events = d.get("events", [])
        if not events:
            print("No audit events")
            return
        print(f"{'Time':<10} {'Event':<25} {'User':<14} {'IP':<16} {'Method':<8} {'Status':<8} {'Risk':<6}")
        print("-" * 87)
        for e in events:
            ts = time.strftime("%H:%M:%S", time.localtime(e.get("timestamp", 0)))
            print(f"{ts:<10} {e.get('event_type', '-'):<25} {e.get('user_id', '-') or '-':<14} "
                  f"{e.get('ip_address', '-') or '-':<16} {e.get('method', '-') or '-':<8} "
                  f"{e.get('status', '-') or '-':<8} {e.get('risk_score', 0):.1f}")
            details = e.get("details")
            if details and args.verbose:
                print(f"           details: {details}")
    elif args.action == "alerts":
        d = _get("/api/security/alerts")
        if _err(d):
            return
        alerts = d.get("alerts", [])
        if not alerts:
            print("No security alerts")
            return
        print(f"{'Time':<10} {'Event':<25} {'User':<14} {'Risk':<8}")
        print("-" * 57)
        for a in alerts:
            ts = time.strftime("%H:%M:%S", time.localtime(a.get("timestamp", 0)))
            print(f"{ts:<10} {a.get('event_type', '-'):<25} {a.get('user_id', '-') or '-':<14} {a.get('risk_score', 0):.1f}")
            details = a.get("details")
            if details:
                print(f"           details: {details}")


def cmd_models(args):
    if args.action == "list":
        query_parts = []
        if args.size_class:
            query_parts.append(f"size_class={args.size_class}")
        if args.min_vram:
            query_parts.append(f"min_vram={args.min_vram}")
        qs = "&".join(query_parts)
        path = "/api/models/catalog" + (f"?{qs}" if qs else "")
        d = _get(path)
        if _err(d):
            return
        models = d.get("models", [])
        if not models:
            print("No models found")
            return
        print(f"  Catalog: {d.get('total_models', 0)} models")
        print(f"{'ID':<25} {'Name':<25} {'Class':<8} {'Params':<10} {'VRAM (MB)':<12} {'License':<12}")
        print("-" * 92)
        for m in models:
            vram = m.get("vram_required_mb", {})
            min_vram = min(vram.values()) if vram else 0
            print(f"{m.get('model_id', '-'):<25} {m.get('name', '-')[:25]:<25} "
                  f"{m.get('size_class', '-'):<8} {m.get('params_m', 0):<10.0f} "
                  f"{min_vram:<12} {m.get('license', '-'):<12}")
    elif args.action == "get":
        if not args.model:
            print("Error: --model required")
            return
        d = _get(f"/api/models/{args.model}")
        if _err(d):
            return
        print(f"  Model:       {d.get('model_id', '-')}")
        print(f"  Name:        {d.get('name', '-')}")
        print(f"  Architecture:{d.get('architecture', '-')}")
        print(f"  Size class:  {d.get('size_class', '-')}")
        print(f"  Parameters:  {d.get('params_m', 0):.0f}M")
        print(f"  Hidden:      {d.get('hidden_size', 0)}")
        print(f"  Layers:      {d.get('num_layers', 0)}")
        print(f"  Heads:       {d.get('num_heads', 0)}")
        print(f"  Vocab:       {d.get('vocab_size', 0)}")
        print(f"  Context:     {d.get('context_length', 0)}")
        print(f"  Quantizations: {', '.join(d.get('quantizations', []))}")
        print(f"  VRAM:        {d.get('vram_required_mb', {})}")
        print(f"  License:     {d.get('license', '-')}")
        print(f"  HuggingFace: {d.get('huggingface_id', '-')}")
        print(f"  Can fit:     {'yes' if d.get('can_fit_available_vram') else 'no'}")
    elif args.action == "vote":
        if not args.model:
            print("Error: --model required for voting")
            return
        data = {"votes": {args.model: args.weight or 1.0}}
        r = _post("/api/autoloader/load", data)
        if _err(r):
            return
        plan = r.get("plan", [])
        print(f"  Vote cast for {args.model} (weight: {args.weight or 1.0})")
        print(f"  Load plan ({len(plan)} models):")
        for p in plan:
            print(f"    {p.get('model_id', '-')} ({p.get('size_class', '-')}) "
                  f"- {p.get('vram_mb', 0):.0f} MB - priority {p.get('priority', 0)}")


def cmd_autoloader(args):
    if args.action == "status":
        d = _get("/api/autoloader/status")
        if _err(d):
            return
        print(f"  Available VRAM:  {d.get('available_vram_mb', 0):.0f} MB")
        print(f"  Available nodes: {d.get('available_nodes', 0)}")
        print(f"  Preferred quant: {d.get('preferred_quant', '-')}")
        print(f"  Max concurrent:   {d.get('max_concurrent_models', 0)}")
        print(f"  Catalog size:     {d.get('catalog_size', 0)}")
        loaded = d.get("loaded_models", [])
        if loaded:
            print(f"\n  Loaded models ({len(loaded)}):")
            for m in loaded:
                print(f"    {m.get('model_id', '-')} ({m.get('size_class', '-')}) - {m.get('vram_mb', 0):.0f} MB")
        recommended = d.get("recommended_loads", [])
        if recommended:
            print(f"\n  Recommended loads ({len(recommended)}):")
            for r in recommended:
                print(f"    {r.get('model_id', '-')} ({r.get('size_class', '-')}) "
                      f"- {r.get('vram_mb', 0):.0f} MB - priority {r.get('priority', 0)}")
    elif args.action == "load":
        data = {}
        if args.model:
            data["force_models"] = [args.model]
        if args.vote:
            data["votes"] = args.vote
        r = _post("/api/autoloader/load", data)
        if _err(r):
            return
        plan = r.get("plan", [])
        print(f"  Load plan ({len(plan)} models):")
        for p in plan:
            print(f"    #{p.get('priority', 0)} {p.get('model_id', '-')} ({p.get('size_class', '-')}) "
                  f"- {p.get('vram_mb', 0):.0f} MB - {p.get('quant', '-')}")
    elif args.action == "recommend":
        params = ""
        if args.vram:
            params = f"?vram_mb={args.vram}"
        d = _get(f"/api/autoloader/recommend{params}")
        if _err(d):
            return
        print(f"  Available VRAM: {d.get('available_vram_mb', 0):.0f} MB")
        models = d.get("recommended_models", [])
        if not models:
            print("  No models fit in available VRAM")
            return
        for m in models:
            print(f"    {m.get('model_id', '-'):<25} {m.get('name', '-')[:25]:<25} "
                  f"{m.get('size_class', '-'):<8} {m.get('vram_mb', 0):<10.0f}")


def cmd_resources(args):
    if args.cluster:
        d = _get("/api/resources/cluster")
        if _err(d):
            return
        print(f"  Cluster CPU:     {d.get('total_cpu_cores', 0)} cores")
        print(f"  Cluster GPU:     {d.get('total_gpu_count', 0)}")
        print(f"  Cluster RAM:     {d.get('total_ram_gb', 0):.0f} GB")
        print(f"  Cluster VRAM:    {d.get('total_vram_mb', 0)} MB")
        print(f"  Contributors:    {d.get('num_contributors', d.get('num_contributers', 0))}")
    else:
        d = _get("/api/resources")
        if _err(d):
            return
        print(f"  CPU:     {d.get('cpu_available', 0)}/{d.get('cpu_cores', 0)} cores")
        print(f"  GPU:     {d.get('gpu_count', 0)} ({', '.join(d.get('gpu_names', [])) or 'none'})")
        vram = d.get('gpu_vram_mb', [])
        avail_vram = d.get('gpu_available_vram_mb', [])
        if vram:
            print(f"  VRAM:    {avail_vram}/{vram} MB")
        print(f"  RAM:     {d.get('ram_available_gb', 0):.1f}/{d.get('ram_total_gb', 0):.1f} GB")
        print(f"  CUDA:    {'yes' if d.get('has_cuda') else 'no'} | ROCm: {'yes' if d.get('has_rocm') else 'no'} | Vulkan: {'yes' if d.get('has_vulkan') else 'no'}")
        print(f"  Score:   {d.get('capacity_score', 0):.1f}")
        print(f"  Summary: {d.get('summary', '')}")


def cmd_leaderboard(args):
    d = _get("/api/pledge/leaderboard")
    if _err(d):
        return
    lb = d.get("leaderboard", [])
    if not lb:
        print("No pledges yet")
        return
    print(f"{'Rank':<6} {'User':<20} {'Score':<10} {'CPU':<6} {'GPU':<6} {'RAM':<8} {'Reputation':<12}")
    print("-" * 68)
    for l in lb:
        print(f"#{l.get('rank', 0):<5} {l.get('user_id', '-'):<20} {l.get('score', 0):<10.1f} "
              f"{l.get('cpu', 0):<6} {l.get('gpu', 0):<6} {l.get('ram_gb', 0):<8.0f} {l.get('reputation', 0):<12.1f}")


def cmd_serve(args):
    import uvicorn
    from netai.api.app import create_app
    from netai.p2p.network import P2PNode

    async def run():
        node = P2PNode(host="0.0.0.0", port=args.p2p_port or 7999,
                       seed_nodes=args.seed or [])
        await node.start()
        app = create_app(p2p_node=node)
        config = uvicorn.Config(app, host=args.host or "0.0.0.0",
                                port=args.port or 8001, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(run())


def main():
    global _args_url
    parser = argparse.ArgumentParser(prog="netai", description="NetAI - Distributed AI Training & Inference")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--url", default=BASE_URL, help="API base URL")
    sub = parser.add_subparsers(dest="command")

    p_status = sub.add_parser("status", help="Show cluster status")
    p_peers = sub.add_parser("peers", help="List connected peers")

    p_res = sub.add_parser("resources", help="Show local or cluster resources")
    p_res.add_argument("--cluster", action="store_true", help="Show cluster-wide resources")

    p_train = sub.add_parser("train", help="Submit training job")
    p_train.add_argument("--model", default="gpt2-small")
    p_train.add_argument("--arch", default="transformer")
    p_train.add_argument("--hidden", type=int, default=768)
    p_train.add_argument("--layers", type=int, default=12)
    p_train.add_argument("--steps", type=int, default=1000)
    p_train.add_argument("--batch", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--device", default="cuda")
    p_train.add_argument("--group", default="")
    p_train.add_argument("--start", action="store_true", help="Start immediately")
    p_train.add_argument("--watch", action="store_true", help="Watch progress")

    p_jobs = sub.add_parser("jobs", help="List training jobs")

    p_vote = sub.add_parser("vote", help="Voting system")
    p_vote.add_argument("action", choices=["propose", "cast", "list"])
    p_vote.add_argument("--model", default="")
    p_vote.add_argument("--arch", default="")
    p_vote.add_argument("--desc", default="")
    p_vote.add_argument("--proposal", default="")
    p_vote.add_argument("--choice", default="for", choices=["for", "against"])
    p_vote.add_argument("--user", default="")
    p_vote.add_argument("--group", default="")

    p_pledge = sub.add_parser("pledge", help="Pledge resources")
    p_pledge.add_argument("--user", default="")
    p_pledge.add_argument("--node", default="")
    p_pledge.add_argument("--cpu", type=int, default=0)
    p_pledge.add_argument("--gpu", type=int, default=0)
    p_pledge.add_argument("--ram", type=float, default=0)
    p_pledge.add_argument("--hours", type=float, default=24)
    p_pledge.add_argument("--group", default="")

    p_lb = sub.add_parser("leaderboard", help="Pledge leaderboard")

    p_group = sub.add_parser("group", help="Group management")
    p_group.add_argument("action", choices=["create", "list", "info", "join", "invite", "propose-training"])
    p_group.add_argument("--name", default="")
    p_group.add_argument("--owner", default="")
    p_group.add_argument("--desc", default="")
    p_group.add_argument("--visibility", default="private", choices=["public", "private", "secret"])
    p_group.add_argument("--max-members", type=int, default=100)
    p_group.add_argument("--approval", default="true")
    p_group.add_argument("--passphrase", default="")
    p_group.add_argument("--group-id", default="")
    p_group.add_argument("--inviter", default="")
    p_group.add_argument("--proposer", default="")
    p_group.add_argument("--model", default="")
    p_group.add_argument("--steps", type=int, default=0)
    p_group.add_argument("--user", default="")
    p_group.add_argument("--invite", default="")
    p_group.add_argument("--cpu", type=int, default=0)
    p_group.add_argument("--gpu", type=int, default=0)
    p_group.add_argument("--ram", type=float, default=0)

    p_inf = sub.add_parser("inference", help="Distributed inference")
    p_inf.add_argument("action", choices=["load", "run", "unload", "status", "models"])
    p_inf.add_argument("--model", default="")
    p_inf.add_argument("--name", default="")
    p_inf.add_argument("--prompt", default="")
    p_inf.add_argument("--max-tokens", type=int, default=256)
    p_inf.add_argument("--temperature", type=float, default=0.7)
    p_inf.add_argument("--top-p", type=float, default=0.9)
    p_inf.add_argument("--version", default="")
    p_inf.add_argument("--shards", type=int, default=1)
    p_inf.add_argument("--device", default="auto")
    p_inf.add_argument("--mirror", default="true")
    p_inf.add_argument("--verbose", action="store_true")

    p_jack = sub.add_parser("jackin", help="Jack into the network")
    p_jack.add_argument("--user", default="")
    p_jack.add_argument("--mode", default="both", choices=["training", "inference", "both"])
    p_jack.add_argument("--cpu", type=int, default=0)
    p_jack.add_argument("--gpu", type=int, default=0)
    p_jack.add_argument("--ram", type=float, default=0)
    p_jack.add_argument("--hours", type=float, default=24)
    p_jack.add_argument("--group", default="")
    p_jack.add_argument("--models", default="", help="Comma-separated models to serve")

    p_grad = sub.add_parser("gradient", help="Gradient sync")
    p_grad.add_argument("action", choices=["status", "push", "pull", "aggregate", "peer", "sync"])
    p_grad.add_argument("--job", default="")
    p_grad.add_argument("--step", type=int, default=0)
    p_grad.add_argument("--peer-id", default="")
    p_grad.add_argument("--endpoint", default="")
    p_grad.add_argument("--node-id", default="")
    p_grad.add_argument("--hash", default="")

    p_auth = sub.add_parser("auth", help="Authentication & authorization")
    p_auth.add_argument("action", choices=["register", "login", "token", "apikey", "verify", "logout", "users"])
    p_auth.add_argument("--user", default="")
    p_auth.add_argument("--password", default="")
    p_auth.add_argument("--role", default="user", choices=["user", "operator", "admin", "node"])
    p_auth.add_argument("--scopes", default="", help="Override default scopes (comma-separated)")
    p_auth.add_argument("--ttl", type=float, default=24.0)
    p_auth.add_argument("--name", default="")
    p_auth.add_argument("--token", default="")

    p_sec = sub.add_parser("security", help="Security status & audit")
    p_sec.add_argument("action", choices=["status", "audit", "alerts"])
    p_sec.add_argument("--limit", type=int, default=20)

    p_models = sub.add_parser("models", help="Model catalog & voting")
    p_models.add_argument("action", choices=["list", "get", "vote"])
    p_models.add_argument("--model", default="")
    p_models.add_argument("--size-class", default="", choices=["mini", "small", "mid", "large"], help="Filter by size class")
    p_models.add_argument("--min-vram", type=float, default=0, help="Filter by minimum available VRAM (MB)")
    p_models.add_argument("--weight", type=float, default=1.0, help="Vote weight (for 'vote' action)")

    p_auto = sub.add_parser("autoloader", help="Auto-loader management")
    p_auto.add_argument("action", choices=["status", "load", "recommend"])
    p_auto.add_argument("--model", default="", help="Force load specific model")
    p_auto.add_argument("--vote", default="", help="Vote JSON: model_id=weight,...")
    p_auto.add_argument("--vram", type=float, default=0, help="Override VRAM for recommendations")

    p_serve = sub.add_parser("serve", help="Start the server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8001)
    p_serve.add_argument("--p2p-port", type=int, default=7999)
    p_serve.add_argument("--seed", nargs="*", default=[])

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _args_url = args.url or BASE_URL

    commands = {
        "status": cmd_status,
        "peers": cmd_peers,
        "train": cmd_train,
        "jobs": cmd_jobs,
        "vote": cmd_vote,
        "pledge": cmd_pledge,
        "leaderboard": cmd_leaderboard,
        "group": cmd_group,
        "inference": cmd_inference,
        "jackin": cmd_jackin,
        "gradient": cmd_gradient,
        "auth": cmd_auth,
        "security": cmd_security,
        "resources": cmd_resources,
        "serve": cmd_serve,
        "models": cmd_models,
        "autoloader": cmd_autoloader,
    }
    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()