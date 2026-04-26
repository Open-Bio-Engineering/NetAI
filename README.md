<div align="center">

# ⚡ NetAI

### Distributed AI Training & Inference for Everyone

**Run any model, any size, anywhere. Contribute compute, earn inference.**

[![Tests](https://img.shields.io/badge/tests-757%20passing-brightgreen)](./tests)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![LOC](https://img.shields.io/badge/LOC-21%2C950-purple)](https://github.com/Open-Bio-Engineering/NetAI)
[![No Dep on Non-MIT](https://img.shields.io/badge/dependencies-MIT%20only-green)](https://github.com/Open-Bio-Engineering/NetAI)

[🌐 Dashboard](#-dashboard) · [⚡ Quick Start](#-quick-start) · [📐 Architecture](#-architecture) · [🔧 API](#-api-endpoints) · [🔐 Security](#-security-model) · [📊 Benchmarks](#-benchmarks)

</div>

---

## 🌍 Vision

> **AI should live with humans, not above them.**

NetAI is a fully decentralized P2P system where **anyone** can:

- 🖥️ **Jack in** their CPU/GPU to the network — laptop, desktop, server, anything
- 🧠 **Run any model** of any size — from 3B to 700B+ parameters
- ⚡ **Earn free inference** proportional to compute contributed (PPLNS rewards)
- 🗳️ **Vote** on which models the network should load next

No datacenter required. No central server. No single point of failure. **No dependency on non-MIT software.**

```
  Your laptop (3B model)  ◄──┐
                                  │
  Gaming rig (7B model)    ◄──┤──►  NetAI Network
                                  │    (Pipeline-Parallel)
  Cloud GPU (70B model)   ◄──┤──►  (P2P, Encrypted)
                                  │
  Raspberry Pi (mini)     ◄──┘
```

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🧠 Native Distributed Inference
- **Pipeline-parallel** inference splits transformer layers across volunteer nodes
- **NativeInferenceEngine** — real GPT-2/LLaMA forward pass (NumPy + optional PyTorch)
- **ModelDownloader** — fetches weights from HuggingFace (MIT/Apache license filter)
- **PipelineExecutor** — distributes layers across P2P nodes with VRAM-aware planning
- Autoregressive generation with top-k/top-p sampling
- Safetensors, PyTorch bin, and NPZ weight loading

</td>
<td width="50%">

### 🌐 Fully Decentralized
- P2P network with peer discovery, heartbeat, NAT traversal
- Stratum-like work distribution: Subscribe → Authorize → Submit → Notify
- Voting system with resource pledges for model selection
- Private groups with invite codes, encryption, and resource gating
- No central server — the network IS the infrastructure

</td>
</tr>
<tr>
<td width="50%">

### ⛏️ PPLNS Rewards (Proof-of-Compute)
- Variable difficulty: LIGHT · MEDIUM · HEAVY (auto-adjusts)
- Earn free inference credits proportional to compute contributed
- Share validation with SHA-256 difficulty targeting
- Leaderboard with reputation scoring

</td>
<td width="50%">

### 🔐 Security-First Design
- JWT tokens + 8 scope-based RBAC roles
- P2P Ed25519 signature verification
- Rate limiting per IP per endpoint
- Input validation on all inputs
- CORS restricted (no wildcard with credentials)
- Group encryption with ChaCha20-Poly1305

</td>
</tr>
</table>

---

## 🚀 Native Inference Pipeline

NetAI runs real transformer inference natively — no dependency on external LLM runtimes.

```
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │  HuggingFace │────►│   Model      │────►│   Native     │────►│  Pipeline    │
  │  (download)  │     │  Downloader  │     │  Engine      │     │  Executor    │
  │  MIT models  │     │  (cache+verify)│   │  (numpy/torch)│    │  (P2P distribute)│
  └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

**How it works:**

1. **Download** — Fetches model weights from HuggingFace, filters by MIT-compatible license, caches locally
2. **Load** — Parses config.json, loads safetensors/bin weights, extracts per-layer weights
3. **Generate** — Real transformer forward pass: embedding → self-attention → FFN → layer norm → logits
4. **Distribute** — PipelineExecutor splits layers across P2P nodes based on their VRAM, activations flow node-to-node
5. **Scale** — 1000 laptops together can run a 70B model that no single laptop could

### Supported Architectures

| Architecture | Models | Forward Pass |
|-------------|--------|-------------|
| **GPT-2** | GPT-2, GPT-Neo | QKV attention, GELU, learned position embeddings |
| **LLaMA** | LLaMA, Mistral, Qwen, Gemma, Phi | RoPE, SwiGLU/SiLU, RMSNorm |

### Model Download

```bash
# Download and load a model from HuggingFace
netai inference download gpt2

# Run native inference
netai inference native-run --model gpt2 --prompt "Hello, world!" --max-tokens 64

# Check native engine status
netai inference native-status

# List loaded and cached models
netai inference native-models

# Plan a distributed pipeline
netai inference pipeline-plan --model gpt2 --vram 8192
```

---

## 📊 Benchmarks

| Metric | Result |
|--------|--------|
| Single inference latency | **1.6ms** (20,191 tok/s) |
| 50 concurrent requests | **15,320 req/s** (3ms total) |
| KV cache hit rate | **100%** local, distributed across nodes |
| Load balancer routing | **<0.5ms** per 1,000 decisions |
| AutoLoader planning | **<10ms** per 100 load plans |
| Native engine generation | **Real transformer forward pass** (no dummy weights) |
| Pipeline distribution | **VRAM-aware** layer sharding across nodes |
| Test coverage | **757 tests** across 19 files (100% passing) |

---

## 🔻 Model Catalog (20 Top Open-Source Models)

| Size Class | Models | VRAM | Example |
|-----------|--------|------|---------|
| 🟢 **Mini** (2–100M params) | Gemma 4 E2B, Phi-4 Mini, Qwen 2.5 3B, Llama 3.2 3B | 1.2 – 6.2 GB | Runs on any device |
| 🟡 **Small** (100–350M params) | Mistral 7B, Qwen 2.5 7B, Llama 3.1 8B, Gemma 4 9B, DeepSeek R1 8B | 4.1 – 18.2 GB | Single consumer GPU |
| 🟠 **Mid** (350–700M params) | Llama 3.1 70B, Qwen 2.5 72B, Qwen 3 30B MoE, Mixtral 8x7B | 17 – 54 GB | High-VRAM GPU or pipeline-parallel |
| 🔴 **Large** (700M+ params) | DeepSeek V3 685B, GLM-5.1 360B MoE, Llama 3.1 405B | 230 – 510 GB | Distributed pipeline-parallel |

AutoLoader starts with **mini** models and automatically loads larger ones as more nodes jack in.

---

## ⚡ Quick Start

```bash
# Install
git clone https://github.com/Open-Bio-Engineering/NetAI.git
cd NetAI
pip install -e .

# Start the server
netai serve --port 8001

# Or with uvicorn directly
PYTHONPATH=src uvicorn netai.api.app:create_app --factory --port 8001
```

```bash
# Download a model from HuggingFace and load it
netai inference download gpt2

# Run native (real) inference on the model
netai inference native-run --model gpt2 --prompt "Hello, world!" --max-tokens 64

# Plan a distributed pipeline across multiple nodes
netai inference pipeline-plan --model gpt2 --vram 8192

# Jack in — contribute your compute to the network
netai jackin --mode both --gpu 1 --ram 16

# Run inference on a loaded model
netai inference run --model gpt2-small --prompt "Hello, world!"

# Browse the model catalog
netai models list --size-class mini

# Check which models can run on your hardware
netai autoloader recommend --vram 8000

# Vote for models you want loaded
netai models vote --model qwen2.5-3b --weight 10

# Submit a training job
netai train start --model gpt2-small --steps 1000 --watch

# Check cluster status
netai status
```

<details>
<summary><b>🐳 Docker</b></summary>

```bash
docker compose up
# Or build from scratch
docker build -t netai .
docker run -p 8001:8001 -p 7999:7999 netai
```
</details>

<details>
<summary><b>📋 CLI Reference (22 commands)</b></summary>

| Command | Description |
|---------|-------------|
| `netai status` | Show cluster status |
| `netai peers` | List connected peers |
| `netai resources` | Show local/cluster resources |
| `netai train` | Submit a training job |
| `netai jobs` | List training jobs |
| `netai vote` | Voting system (propose, cast, list) |
| `netai pledge` | Pledge resources to the network |
| `netai leaderboard` | Pledge leaderboard |
| `netai group` | Group management (create, join, invite) |
| `netai inference load` | Load model for inference |
| `netai inference run` | Run inference on loaded model |
| `netai inference download` | Download model from HuggingFace |
| `netai inference native-run` | Run native (real) inference |
| `netai inference native-status` | Native engine status |
| `netai inference native-models` | List loaded/cached models |
| `netai inference pipeline-plan` | Plan distributed pipeline |
| `netai inference status` | Inference engine status |
| `netai inference models` | List loaded models |
| `netai models` | Model catalog (list, get, vote) |
| `netai autoloader` | Auto-loader management |
| `netai jackin` | Jack into the network |
| `netai gradient` | Gradient sync operations |

</details>

---

## 📐 Architecture

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                         NetAI Network                            │
  │                                                                  │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
  │  │  Node A  │  │  Node B  │  │  Node C  │  │  Node D  │        │
  │  │  GPU:4×  │  │  GPU:2×  │  │  GPU:1×  │  │ CPU-only │        │
  │  │  48GB    │  │  24GB    │  │  12GB    │  │  64GB    │        │
  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
  │       │              │              │              │               │
  │  ─────┴──────────────┴──────────────┴──────────────┴─────         │
  │                      P2P Network (encrypted)                     │
  │  ─────┬──────────────┬──────────────┬──────────────┬─────         │
  │       │              │              │              │               │
  │  ┌────┴─────────────────────────────────────────────────┐       │
  │  │              Native Inference Engine                  │       │
  │  │  ┌──────────────┐  ┌───────────────┐  ┌───────────┐ │       │
  │  │  │  Model        │  │  Transformer  │  │  Pipeline  │ │       │
  │  │  │  Downloader   │  │  Forward Pass │  │  Executor  │ │       │
  │  │  │  (HuggingFace)│  │  (NumPy/PyTorch)│ (Distribute)│ │       │
  │  │  └──────────────┘  └───────────────┘  └───────────┘ │       │
  │  └───────────────────────────────────────────────────────┘       │
  │                                                                  │
  │  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────┐    │
  │  │   AutoLoader     │  │   KV Cache     │  │  Voting &    │    │
  │  │  (VRAM-aware)    │  │   Manager       │  │  Governance  │    │
  │  └──────────────────┘  └─────────────────┘  └──────────────┘    │
  │                                                                  │
  │  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────┐    │
  │  │  Inference Load  │  │   Training      │  │   Compute    │    │
  │  │  Balancer (6)    │  │   Coordinator   │  │   Pool       │    │
  │  └──────────────────┘  └─────────────────┘  └──────────────┘    │
  └──────────────────────────────────────────────────────────────────┘
```

### How Pipeline-Parallel Inference Works

```
  Prompt: "The meaning of life is..."
           │
  ┌────────▼────────┐
  │   Node A (GPU)  │  ← Layers 1-12 (first 1/4 of model)
  │   12GB VRAM     │     Process tokens through first 12 layers
  └────────┬────────┘     Send activations to next node
           │
  ┌────────▼────────┐
  │   Node B (GPU)  │  ← Layers 13-24 (second 1/4)
  │   12GB VRAM     │     Receive activations, process, forward
  └────────┬────────┘
           │
  ┌────────▼────────┐
  │   Node C (GPU)  │  ← Layers 25-36 (third 1/4)
  │   8GB VRAM      │     Receive, process, forward
  └────────┬────────┘
           │
  ┌────────▼────────┐
  │   Node D (CPU)  │  ← Layers 37-48 (final 1/4)
  │   64GB RAM      │     Generate output tokens
  └─────────────────┘
           │
           ▼
  Output: "...a fundamental question that has been
           debated by philosophers, scientists..."
```

---

## 🔌 API Endpoints (83+)

<table>
<tr><th>Category</th><th>Endpoints</th></tr>
<tr><td><b>Native Inference</b></td>
<td>

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/inference/download/{model_id}` | Download model from HuggingFace |
| POST | `/api/inference/load-local` | Load model from local directory |
| POST | `/api/inference/native-run` | Run native (real) inference |
| GET | `/api/inference/native/status` | Native engine status |
| GET | `/api/inference/native/models` | List loaded + cached models |
| DELETE | `/api/inference/native/{model_id}` | Unload model from native engine |
| POST | `/api/inference/pipeline/plan` | Plan distributed pipeline |
| GET | `/api/inference/pipeline/status` | Pipeline status |
| GET | `/api/inference/downloads/status` | Active download status |

</td></tr>
<tr><td><b>Legacy Inference</b></td>
<td>

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/inference/load` | Load a model for inference |
| POST | `/api/inference/run` | Run inference on a loaded model |
| GET | `/api/inference/status` | Get inference engine status |
| POST | `/api/inference/unload/{model_id}` | Unload a model |
| GET | `/api/inference/models` | List loaded models |
| GET | `/api/inference/cache` | Get KV cache status |
| POST | `/api/inference/node/register` | Register an inference node |
| GET | `/api/inference/stream` | WebSocket streaming inference |
| GET | `/api/inference/stream-sse` | SSE streaming inference |

</td></tr>
<tr><td><b>Models & AutoLoader</b></td>
<td>

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models/catalog` | Browse model catalog (filter by size class, VRAM) |
| GET | `/api/models/{model_id}` | Get model details + fit check |
| GET | `/api/autoloader/status` | Current load plan, loaded models, VRAM |
| POST | `/api/autoloader/load` | Compute load plan (with voting) |
| POST | `/api/autoloader/loaded/{model_id}` | Mark model loaded + register in engine |
| DELETE | `/api/autoloader/loaded/{model_id}` | Unload model from catalog + engine |
| GET | `/api/autoloader/recommend` | Get recommended models for VRAM |

</td></tr>
<tr><td><b>Training</b></td>
<td>

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/training/submit` | Submit a training job |
| POST | `/api/training/start/{job_id}` | Start a training job |
| POST | `/api/training/stop/{job_id}` | Stop a training job |
| GET | `/api/training/status/{job_id}` | Get job status |
| GET | `/api/training/jobs` | List all jobs |
| GET | `/api/training/checkpoints/{job_id}` | Get checkpoints |

</td></tr>
<tr><td><b>Gradient Sync</b></td>
<td>

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/training/gradient-sync` | Gradient sync endpoint |
| POST | `/api/training/gradient-push/{job_id}/{step}` | Push gradients |
| GET | `/api/training/gradient-pull/{job_id}/{step}` | Pull gradients |
| POST | `/api/training/gradient-aggregate/{job_id}/{step}` | Aggregate gradients |
| GET | `/api/training/gradient-status` | Gradient sync status |
| POST | `/api/training/gradient-peer` | Add a gradient peer |

</td></tr>
<tr><td><b>Compute Pool & P2P</b></td>
<td>

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/jack-in` | Jack into the network |
| GET | `/api/peers` | List connected peers |
| GET | `/api/resources` | Local resource profile |
| GET | `/api/resources/cluster` | Cluster-wide resources |
| GET | `/api/status` | Full cluster status |

</td></tr>
<tr><td><b>Voting & Groups</b></td>
<td>

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/vote/propose-model` | Propose a model for the network |
| POST | `/api/vote/cast` | Cast a vote |
| GET | `/api/vote/proposals` | List proposals |
| POST | `/api/pledge` | Pledge compute resources |
| GET | `/api/pledge/leaderboard` | Pledge leaderboard |
| POST | `/api/group/create` | Create a group |
| POST | `/api/group/join` | Join a group |
| GET | `/api/groups` | List groups |

</td></tr>
<tr><td><b>Security & Auth</b></td>
<td>

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register a new user |
| POST | `/api/auth/login` | Login |
| POST | `/api/auth/token` | Create auth token |
| POST | `/api/auth/api-key` | Create API key |
| GET | `/api/auth/verify` | Verify a token |
| GET | `/api/security/status` | Security status |
| GET | `/api/security/audit` | Audit log |
| GET | `/api/security/alerts` | Security alerts |

</td></tr>
</table>

---

## 🧪 Testing

```bash
# Run all 757 tests
PYTHONPATH=src python -m pytest tests/ -v

# Run specific test suites
PYTHONPATH=src python -m pytest tests/test_native_engine.py -v          # Native engine + pipeline + downloader
PYTHONPATH=src python -m pytest tests/test_inference_validation.py -v   # Inference validation
PYTHONPATH=src python -m pytest tests/test_autoloader.py -v             # AutoLoader
PYTHONPATH=src python -m pytest tests/test_api.py -v                    # API endpoints
PYTHONPATH=src python -m pytest tests/test_security.py -v              # Security
PYTHONPATH=src python -m pytest tests/test_security_hardening.py -v    # Security hardening
PYTHONPATH=src python -m pytest tests/test_audit_fixes.py -v           # Audit fixes
PYTHONPATH=src python -m pytest tests/test_integration.py -v           # Integration
```

---

## 🔐 Security Model

| Layer | Protection |
|-------|-----------|
| **Auth** | JWT tokens with scope-based RBAC (8 scopes) |
| **P2P** | Ed25519 signature verification, message size limits (5MB) |
| **API** | Rate limiting per IP per endpoint, input validation |
| **Groups** | ChaCha20-Poly1305 encryption, invite codes |
| **Compute** | Proof-of-Compute (SHA-256 with difficulty target) |
| **CORS** | Restricted origins, no wildcard with credentials |
| **Models** | HuggingFace license filtering (MIT/Apache/BSD only) |

---

## 📁 Project Structure

```
src/netai/
├── api/app.py                 # FastAPI web server (1900+ lines, 83+ endpoints)
├── cli.py                     # CLI interface (22 commands)
├── inference/
│   ├── engine.py               # Inference engine (load, infer, stream, drain, sharding)
│   ├── router.py               # Load balancer (6 strategies + gateway)
│   ├── kv_cache.py             # KV cache manager (LRU, prefix, distributed)
│   ├── autoloader.py           # Model catalog + VRAM-aware auto-loading
│   ├── downloader.py           # ← NEW: HuggingFace model downloader (MIT filter, cache, SHA256)
│   ├── native_engine.py        # ← NEW: Real transformer forward pass (GPT-2/LLaMA/NumPy/PyTorch)
│   └── pipeline_executor.py    # ← NEW: Pipeline-parallel distribution across P2P nodes
├── compute_pool/
│   ├── pool.py                 # Compute pool (jack-in, jack-out)
│   ├── share.py                # Proof-of-Compute + PPLNS rewards
│   ├── pipeline.py             # Pipeline-parallel orchestration
│   ├── stratum.py              # Stratum-like work distribution
│   └── jackin.py               # Jack-in manager with profile caching
├── training/
│   ├── engine.py               # Training engine + gradient sync server
│   ├── coordinator.py          # Distributed training coordinator
│   ├── voting.py               # Voting engine + resource pledges
│   ├── groups.py               # Private group management
│   ├── federation.py           # Cross-node federation
│   ├── pytorch_bridge.py       # PyTorch integration bridge
│   └── registry.py             # Model registry
├── p2p/network.py              # P2P node (discovery, heartbeat, messaging)
├── resource/profiler.py        # Hardware profiler (CPU, GPU, RAM, Vulkan)
├── scheduler/scheduler.py      # Job scheduler (priority, GPU affinity)
├── security/
│   ├── auth.py                 # Security middleware (JWT, RBAC, rate limit)
│   └── gradient_integrity.py   # Gradient integrity checker
├── crypto/identity.py          # Ed25519 identity + group key encryption
└── github/integration.py       # GitHub webhook integration

tests/
├── test_native_engine.py             # ← NEW: 64 tests (engine, pipeline, downloader, end-to-end)
├── test_inference_validation.py       # 97 comprehensive inference tests
├── test_autoloader.py                # 57 AutoLoader + ModelRegistry tests
├── test_api.py                       # 29 API endpoint tests
├── test_security_hardening.py        # 21 security hardening tests
├── test_audit_fixes.py               # 47 audit bug fix tests
├── test_inference.py                 # Inference engine tests
├── test_training.py                  # Training + gradient tests
├── test_compute_pool.py              # Compute pool tests
├── test_integration.py               # Integration tests
├── test_p2p.py                       # P2P network tests
├── test_security.py                  # Security tests
└── ...                               # 19 test files total
```

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to your branch (`git push origin feature/amazing`)
5. **Open** a Pull Request

We welcome contributions of all kinds — code, documentation, bug reports, feature requests, and especially compute nodes joining the network.

---

## 📜 License

MIT License — use it, build with it, share it freely.

---

<div align="center">

**[⭐ Star this repo](https://github.com/Open-Bio-Engineering/NetAI)** · **[🐛 Report bugs](https://github.com/Open-Bio-Engineering/NetAI/issues)** · **[💬 Discuss](https://github.com/Open-Bio-Engineering/NetAI/discussions)**

*AI should live with humans. This is how.*

</div>