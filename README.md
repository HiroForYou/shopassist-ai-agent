# shopassist-agentic

Agente de soporte de reembolsos para una tienda online, construido por fases con LLM local:
tool calling, multi-agent, RAG, evaluacion, observabilidad, guardrails y streaming.

**Dominio:** ShopAssist. Reglas de negocio deterministas (plazo, categoria, monto) que permiten medir si el agente acierta.

**Stack:** Ollama (LLM local) · LangChain · LangGraph · LangSmith · Qdrant · Docker · pytest.

## Requisitos

- Docker Desktop (o Python 3.11+ para ejecucion local).
- ~8 GB de RAM libres para los modelos de Ollama (CPU funciona; GPU NVIDIA opcional, ver [Fase 0](docs/fase-00-setup.md)).
- Cuenta gratuita en [LangSmith](https://smith.langchain.com) para trazas y experimentos.

## Roadmap

| Fase | Tema | Concepto que practica | Estado |
|---|---|---|---|
| 0 | [Setup](docs/fase-00-setup.md) | Docker, Ollama, LangSmith, healthcheck | Listo |
| 1 | [Agente con tools](docs/fase-01-agente-tools.md) | Loop agentico manual, tool calling, errores de tools | Listo |
| 2 | [Multi-agent con LangGraph](docs/fase-02-multiagent.md) | Router, especialistas, handoffs, estado compartido, memoria (checkpointer) | Listo |
| 3 | [RAG](docs/fase-03-rag.md) | Politicas en Qdrant, embeddings Ollama, citas, eval de retrieval (hit@k, MRR) | Listo |
| 4 | [Evaluacion](docs/fase-04-evaluacion.md) | Datasets en LangSmith, experimentos, LLM-as-judge calibrado, consistencia, gate de regresion | Listo |
| 5 | [Observabilidad](docs/fase-05-observabilidad.md) | Logs JSON por evento, metricas p50/p95, costo, router hibrido, runbook de debugging | Listo |
| 6 | Guardrails | Validacion input/output, permisos por tool, human-in-the-loop, fallback de modelo | Pendiente |
| 7 | Streaming | FastAPI + SSE, time-to-first-token, metricas de latencia | Pendiente |

## Estructura

```
shopassist-agentic/
├── docker-compose.yml     # ollama + ollama-pull + qdrant + app
├── knowledge/             # politicas de la tienda (base del RAG)
├── Dockerfile             # imagen de la app (python 3.11)
├── .env.example           # copiar a .env
├── src/agentic/
│   ├── config.py          # settings (.env)
│   ├── llm.py             # factory de ChatOllama
│   ├── agent.py           # Fase 1: loop agentico
│   ├── multiagent.py      # Fase 2-3: grafo LangGraph (router + 3 especialistas)
│   ├── rag.py             # Fase 3: chunking, embeddings, Qdrant, tool search_policies
│   ├── tools.py           # tools del dominio
│   ├── evalkit.py         # utilidades de evaluacion (flujo, checks, metricas, reportes)
│   ├── graph_eval.py      # runner de casos multi-turno contra el grafo
│   ├── experiments.py     # Fase 4: datasets, target, evaluadores, regresiones
│   ├── judges.py          # Fase 4: LLM-as-judge (groundedness, policy_compliance)
│   ├── facts.py           # Fase 4: hechos verificables por codigo (R2/R3)
│   ├── observability.py   # Fase 5: logs JSON, callback por nodo, costo
│   └── domain/            # pedidos, politica de reembolso
├── evals/                 # casos de evaluacion por fase (JSON)
├── scripts/               # entrypoint por fase (chat, eval, healthcheck)
├── reports/               # reportes JSON de evaluacion (ignorado por git)
├── logs/                  # logs JSON de observabilidad (ignorado por git)
├── tests/                 # tests sin LLM (deterministas)
└── docs/                  # guia de cada fase
```

## Inicio rapido

```powershell
git clone https://github.com/<usuario>/shopassist-agentic.git
cd shopassist-agentic
Copy-Item .env.example .env        # completar LANGSMITH_API_KEY
docker compose up -d ollama qdrant
docker compose run --rm ollama-pull
docker compose run --rm app python scripts/fase3_ingest.py
docker compose run --rm app python scripts/fase0_healthcheck.py
docker compose run --rm app python scripts/fase1_agent.py
docker compose run --rm app pytest -q
```

Ejecucion local con `env_ml` (Ollama igual en Docker): ver [Fase 0](docs/fase-00-setup.md#opcion-b-entorno-local-env_ml).
