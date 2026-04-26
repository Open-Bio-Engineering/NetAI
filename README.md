# NetAI — Distributed AI Training & Inference for Everyone

**NetAI** is an open-source distributed AI system where anyone can contribute compute (CPU/GPU) and run models of any size — from 3B to 700B+ parameters — without a datacenter. Pipeline-parallel inference splits transformer layers across volunteer nodes. PPLNS rewards give contributors free inference credits proportional to their compute.

## Features

- **Pipeline-Parallel Inference** — Split any transformer model across volunteer nodes; no single GPU needed
- **Distributed Training** — Gradient sync, shard scheduling, checkpoint management
- **P2P Network** — Decentralized peer discovery, heartbeat, NAT traversal
- **Voting & Governance** — Resource pledges, weighted voting, model selection by community
- **Private Groups** — Invite-only training groups with resource gating
- **PPLNS Rewards** — Proof-of-Compute with variable difficulty (LIGHT/MEDIUM/HEAVY)
- **Stratum Protocol** — Subscribe → Authorize → Submit → Notify work distribution
- **KV Cache** — Distributed key-value cache with partition affinity
- **Security** — JWT auth, scope-based RBAC, rate limiting, input validation, P2P signature verification
- **Inference Gateway** — Load balancing (round-robin, least-loaded, hash-based, adaptive), mirroring
- **CLI** — Full command-line interface for all operations

## Quick Start

```bash
# Install
pip install -e .

# Start server
netai serve --port 8001

# Or with uvicorn
PYTHONPATH=src uvicorn netai.api.app:create_app --factory --port 8001

# Open dashboard
open http://localhost:8001

# Jack in (contribute compute)
netai jack-in --mode both --model gpt2-small

# Run inference
netai infer run --model gpt2-small --prompt "Hello, world"

# Train a model
netai train start --model gpt2-small --steps 1000
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Volunteer   │────▶│  Load Balancer│────▶│   Replica    │
│  Node (GPU)  │     │  + Gateway   │     │   Manager    │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                     │
       │           ┌───────┴───────┐    ┌───────┴───────┐
       │           │  KV Cache     │    │  Pipeline      │
       │           │  Manager      │    │  Orchestrator  │
       │           └───────────────┘    └───────┬───────┘
       │                                         │
       │    ┌─────────────┐             ┌───────┴───────┐
       └───▶│  P2P Network │             │  Compute Pool  │
            │  + Voting    │             │  + Stratum     │
            └─────────────┘             └───────────────┘
```

## API Endpoints

45+ endpoints covering:

- **Inference**: load, run, unload, cache, stream (WebSocket/SSE), mirror
- **Training**: submit, start, stop, status, checkpoints
- **Gradient Sync**: push, pull, aggregate, peer, status
- **Compute Pool**: jack-in, jack-out, subscribe, submit, leaderboard
- **Voting**: propose, cast, proposals, leaderboard
- **Groups**: create, join, leave, invite, propose training
- **Auth**: register, login, tokens, API keys
- **P2P**: join, leave, heartbeat, message

## Testing

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

## Docker

```bash
docker compose up
```

## Security Model

- JWT tokens with scope-based RBAC (8 scopes: READ, WRITE, TRAIN, INFERENCE, GRADIENT, VOTE, GROUP, ADMIN)
- P2P Ed25519 signature verification
- Rate limiting per IP per endpoint
- Input validation on all endpoints
- CORS restricted (no wildcard + credentials)
- Proof-of-Compute for share validation
- Group encryption with ChaCha20-Poly1305

## License

MIT