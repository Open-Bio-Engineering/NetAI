# NetAI — Spread the Word

## Twitter/X Post (280 chars)

🚀 NetAI: an open-source P2P system where ANYONE can contribute CPU/GPU and run AI models of any size (3B→700B+) without a datacenter.

Pipeline-parallel inference splits models across volunteer nodes.
PPLNS rewards = free inference for compute.
693 tests passing.

⭐ https://github.com/Open-Bio-Engineering/NetAI

---

## Twitter/X Thread Version

1/ 🧠 What if AI could live WITH humans instead of above them?

Introducing NetAI — a fully decentralized P2P system where anyone can:
• Jack in their CPU/GPU
• Run models from 3B to 700B+ params
• Earn free inference via PPLNS rewards
• No datacenter needed

2/ ⚡ How it works:

Pipeline-parallel inference splits transformer layers across volunteer nodes. Your laptop runs Layers 1-12, someone's gaming rig runs 13-24, a cloud GPU runs 25-36... any model, any size, distributed across the network.

3/ 🏗️ Architecture:

• Inference engine with 6 routing strategies (round-robin, adaptive, hash-based...)
• AutoLoader that starts with mini models and scales up as nodes join
• KV cache with distributed partitioning
• P2P network with Ed25519 signatures
• Voting system for community model selection

4/ 📊 Benchmarks on a single machine:

• 20,191 tokens/sec inference throughput
• 15,320 req/s concurrent
• <0.5ms load balancer routing
• 693 tests, 100% passing

5/ 🔐 Security-first:

• JWT + 8 scope-based RBAC roles
• P2P Ed25519 signature verification
• Rate limiting, input validation
• ChaCha20-Poly1305 group encryption
• Proof-of-Compute with variable difficulty

6/ 🚀 73+ API endpoints, 17 CLI commands:

netai jack-in --mode both --gpu 1
netai models list --size-class mini
netai autoloader recommend --vram 8000
netai inference run --model gpt2-small --prompt "Hello"

7/ 💚 19,290 lines of Python. 62 source files. MIT licensed.

The network IS the infrastructure. AI should live with humans.

⭐ https://github.com/Open-Bio-Engineering/NetAI

---

## Reddit r/MachineLearning Post

**Title:** NetAI — Open-source P2P system to run any LLM (3B→700B+) by pooling volunteer compute. Pipeline-parallel inference, PPLNS rewards, 693 tests passing.

**Body:**

We built NetAI, a fully decentralized AI system where anyone can contribute compute (CPU/GPU) and run models of any size without a datacenter.

**How it works:**
- Pipeline-parallel inference splits transformer layers across volunteer nodes — your laptop runs the first 12 layers, a gaming rig runs the next 12, etc.
- AutoLoader starts with mini models (Gemma 4 2B) and automatically loads larger models as more nodes join the network
- PPLNS (Pay-Per-Last-N-Shares) rewards give contributors free inference credits proportional to their compute
- Community voting decides which models the network loads next

**Architecture:**
- Inference engine with model serving, sharding, mirroring, streaming (SSE + WebSocket)
- Load balancer with 6 routing strategies (round-robin, least-loaded, lowest-latency, hash-based, random, adaptive)
- Distributed KV cache with partition affinity, prefix caching, LRU eviction
- 20-model catalog across 4 size classes (mini/small/mid/large)
- Stratum-like work distribution with variable difficulty Proof-of-Compute
- P2P network with Ed25519 signatures, heartbeat, NAT traversal

**Benchmarks:**
- 20,191 tokens/sec single inference
- 15,320 req/s concurrent (50 requests)
- <0.5ms routing decision latency

**Tech stack:** Python, FastAPI, async/await, NumPy, Pydantic v2, WebSocket, SSE

**Repo:** https://github.com/Open-Bio-Engineering/NetAI

MIT licensed. Contributions welcome.

---

## Hacker News Post

**Title:** NetAI – Distributed AI inference and training where anyone can contribute compute

We open-sourced NetAI, a P2P system for running AI models of any size by pooling volunteer compute. Uses pipeline-parallel inference to split transformer layers across nodes, with PPLNS rewards for contributors.

- 20K tok/s inference throughput, 15K req/s concurrent
- 693 tests passing, 19K LOC, MIT licensed
- AutoLoader starts with mini models, scales up as nodes join
- 6 load-balancing strategies, distributed KV cache, streaming
- Stratum-like work distribution, voting system, private groups
- Ed25519 P2P signatures, JWT+RBAC auth, encrypted groups

https://github.com/Open-Bio-Engineering/NetAI

---

## Discord/Slack Message

🚀 **NetAI v1.0.0 is live!** — An open-source P2P system where anyone can contribute CPU/GPU and run AI models of any size (3B to 700B+ params) without a datacenter.

✅ Pipeline-parallel inference across volunteer nodes
✅ PPLNS rewards = free inference for compute contributors
✅ AutoLoader: starts mini, scales to large as nodes join
✅ 20K tok/s, 15K req/s, 693 tests passing
✅ 73+ API endpoints, 17 CLI commands
✅ MIT licensed

⭐ https://github.com/Open-Bio-Engineering/NetAI

---

## LinkedIn Post

Excited to share NetAI — an open-source system that makes AI compute accessible to everyone.

The idea is simple: anyone can "jack in" their CPU or GPU to a P2P network, and in return earn inference credits proportional to their contribution. Pipeline-parallel inference splits transformer models across volunteer nodes, so even models with 700B+ parameters can run without a datacenter.

Key features:
• 6 load-balancing strategies including adaptive routing
• AutoLoader that scales from mini to large models based on available VRAM
• Proof-of-Compute with PPLNS rewards
• Full security: JWT auth, P2P signatures, encrypted groups
• 693 tests passing, 19K lines of code

Would love feedback and contributions: https://github.com/Open-Bio-Engineering/NetAI

#OpenSource #AI #DistributedComputing #P2P #MachineLearning