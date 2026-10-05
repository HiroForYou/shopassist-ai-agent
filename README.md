# ShopAssist agentic

**English** · [Español](README.es.md)

Refund-support assistant for an online store, built on a local LLM and organized in eight phases:
tool-using agent, multi-agent, RAG, evaluation, observability, guardrails and streaming.

The business rules (30-day window, digital products, amounts over 200 USD) are deterministic, so every
agent response can be checked against an expected result.

![Runtime architecture](docs/diagramas/_capturas/arquitectura-runtime/arquitectura-runtime.visual-check.1440x900.light.png)

## Stack

| Layer | Technology |
|---|---|
| Local LLM | Ollama: `qwen3.5:4b` (main), `granite4.1:3b` (fallback), `qwen3-embedding:0.6b` (embeddings) |
| Orchestration | LangChain, LangGraph |
| Retrieval | Qdrant |
| Evaluation and tracing | LangSmith |
| API | FastAPI with SSE |
| Metrics | Prometheus, Grafana |
| Infrastructure | Docker Compose on WSL2 |
| Tests | pytest (no LLM) |

## Phases

| Phase | Topic | Content | Key result |
|---|---|---|---|
| 0 | [Setup](docs/fase-00-setup.md) | Docker, Ollama, LangSmith, healthcheck | Reproducible CPU environment |
| 1 | [Tool-using agent](docs/fase-01-agente-tools.md) | Hand-written agent loop, tool calling, errors as data | 11/15 cases |
| 2 | [Multi-agent](docs/fase-02-multiagent.md) | Router, specialists, handoffs, per-conversation memory | 11/12 cases |
| 3 | [RAG](docs/fase-03-rag.md) | Policies in Qdrant, citations, retrieval evaluation | hit@3 = 1.00, MRR = 0.93 |
| 4 | [Evaluation](docs/fase-04-evaluacion.md) | LangSmith datasets, calibrated LLM-as-judge, regression gate | Rule-based judge with 0 FP |
| 5 | [Observability](docs/fase-05-observabilidad.md) | JSON logs, per-node metrics, hybrid router | -20 % latency, -19 % tokens |
| 6 | [Guardrails](docs/fase-06-guardrails.md) | Input, per-tool policies, output, resilience | Refunds without confirmation: 0/9 → 9/9 |
| 7 | [Streaming](docs/fase-07-streaming.md) | SSE API, TTFT, backpressure, Prometheus and Grafana | First progress event in 0.01-0.04 s |

Additional documentation:

| Document | Content |
|---|---|
| [docs/RESULTADOS.md](docs/RESULTADOS.md) | Metrics per phase, design decisions and limitations |
| [docs/diagramas/](docs/diagramas/README.md) | Architecture, refund-flow sequences, lifecycle and evaluation loop |

The phase guides and diagrams are written in Spanish.

## Requirements

| Resource | Minimum |
|---|---|
| Docker Desktop | WSL2 backend (or Python 3.11+ to run locally) |
| RAM for WSL2 | 8 GB (Ollama models on CPU; NVIDIA GPU optional, see [Phase 0](docs/fase-00-setup.md)) |
| Disk | ~7 GB for models and images |
| LangSmith | Free account and API key |

## Quick start

```powershell
Copy-Item .env.example .env                 # fill in LANGSMITH_API_KEY
docker compose up -d ollama qdrant
docker compose run --rm ollama-pull         # ~6 GB the first time
docker compose build app api
docker compose run --rm app python scripts/fase3_ingest.py
docker compose run --rm app python scripts/fase0_healthcheck.py
docker compose run --rm app pytest -q

# console chat with the full system
docker compose run --rm app python scripts/fase2_chat.py

# streaming API and monitoring
docker compose up -d api
docker compose --profile monitoring up -d
docker compose run --rm app python scripts/fase7_client.py chat
```

| Service | URL |
|---|---|
| API (Swagger) | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000/d/shopassist/shopassist-api-y-agentes |
| Qdrant | http://localhost:6333/dashboard |

## Project structure

```
ai-agentic/
├── docker-compose.yml     # ollama, qdrant, app, api; ollama-pull (tools profile); prometheus and grafana (monitoring profile)
├── Dockerfile             # Python 3.11 image for app and api
├── .env.example           # configuration (copy to .env)
├── infra/                 # Prometheus and Grafana dashboard
├── knowledge/             # store policies (RAG source)
├── src/agentic/
│   ├── config.py          # settings loaded from .env
│   ├── llm.py             # ChatOllama, resilient model, model digests
│   ├── agent.py           # Phase 1: agent loop
│   ├── multiagent.py      # Phases 2-7: LangGraph graph, hybrid router, per-turn execution
│   ├── rag.py             # Phase 3: chunking, embeddings, Qdrant, search_policies
│   ├── tools.py           # domain tools
│   ├── evalkit.py         # evaluation checks, metrics and reports
│   ├── graph_eval.py      # multi-turn case runner
│   ├── experiments.py     # Phase 4: datasets, evaluators, regressions
│   ├── judges.py          # Phase 4: LLM-as-judge
│   ├── facts.py           # Phase 4: facts verifiable in code
│   ├── observability.py   # Phase 5: JSON logs and per-node callback
│   ├── guardrails.py      # Phase 6: input, tools, output, resilience
│   ├── api.py             # Phase 7: FastAPI, SSE, backpressure
│   ├── metrics.py         # Phase 7: Prometheus metrics
│   └── domain/            # orders and refund policy
├── evals/                 # evaluation cases (JSON)
├── scripts/               # entry points per phase
├── tests/                 # 156 tests, no LLM
├── reports/               # evaluation reports (not versioned)
├── logs/                  # JSON logs (not versioned)
└── docs/                  # phase guides, results and diagrams
```

## Running the code locally without Docker

Ollama and Qdrant stay in Docker; scripts and tests run in a local Python environment (see
[Phase 0](docs/fase-00-setup.md#opción-b-entorno-python-local)).
