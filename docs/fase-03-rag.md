# Fase 3: RAG

Respuestas sobre políticas de la tienda a partir de una base de conocimiento, con cita de la fuente y sin inventar
cuando la información no existe.

Código: [rag.py](../src/agentic/rag.py) · Base: [knowledge/](../knowledge/) · Agente: `policy_agent` en [multiagent.py](../src/agentic/multiagent.py)

## Pipeline

```
knowledge/*.md ──chunk por "##"──> 23 chunks ──qwen3-embedding:0.6b──> Qdrant (coseno)
                                                                          │
pregunta ──instrucción + embed_query──> búsqueda top-k + umbral ──────────┘──> search_policies ──> policy_agent
```

| Componente | Decisión | Motivo |
|---|---|---|
| Chunking | Una sección `##` por chunk (100-300 caracteres) | Las secciones son unidades semánticas; el ID `archivo#seccion` sirve como cita |
| Contexto del chunk | Se embebe `"<título> - <sección>\n<texto>"` | Una sección aislada pierde el documento al que pertenece |
| Embeddings | `qwen3-embedding:0.6b` vía Ollama | Multilingüe, ~640 MB, corre en CPU; alternativa evaluable `bge-m3` |
| Instrucción en la consulta | `Instruct: ...\nQuery: <pregunta>` solo en consultas | Modelo entrenado de forma asimétrica; efecto medible con `--no-instruction` |
| Base vectorial | Qdrant, coseno, IDs `uuid5(chunk_id)` | Ingesta idempotente: reindexar no duplica puntos |
| Recuperación | `top_k=3`, `score_threshold=0.40` | Umbral calibrado con la evaluación de retrieval |
| Generación | `policy_agent` siempre busca, cita `[fuente]` y responde "no tengo esa información" si no hay respaldo | Control de alucinaciones en el prompt; medido con LLM-as-judge (Fase 4) |
| Trazas | `KnowledgeBase.search` con `@traceable(run_type="retriever")` | LangSmith muestra documentos y scores por búsqueda |

Base de conocimiento: `reembolsos.md`, `pagos.md`, `envios.md`, `garantia.md`, `atencion.md`.

## Integración con el grafo

| Mensaje | Ruta | Quién decide |
|---|---|---|
| "Se puede devolver un e-book" (regla general) | `policies` | RAG |
| "Puedo devolver el A1004" (pedido concreto) | `refunds` | `policy.py` |
| "En cuánto tiempo me devuelven el dinero" durante un reembolso | handoff a `policy_agent` | RAG |

La base documenta las reglas; la decisión sobre un pedido concreto sigue siendo determinista.

## Ejecución

```powershell
docker compose run --rm app python scripts/fase3_ingest.py               # reindexar tras editar knowledge/
docker compose run --rm app python scripts/fase3_eval.py retrieval       # solo embeddings, rápido
docker compose run --rm app python scripts/fase3_eval.py retrieval --no-instruction
docker compose run --rm app python scripts/fase3_eval.py e2e             # sistema completo, 10 casos
docker compose run --rm app pytest -q tests/test_rag.py                  # Qdrant en memoria, sin Ollama
```

## Evaluación de retrieval

Dataset: [fase3_retrieval.json](../evals/fase3_retrieval.json). 22 preguntas parafraseadas con sus chunks esperados y
8 sin respuesta en la base (5 del dominio de la tienda, 3 ajenas).

| Métrica | Definición |
|---|---|
| hit@1 | El chunk esperado es el primero |
| hit@k | El chunk esperado está entre los k recuperados (lo que recibe el LLM) |
| MRR | Media de 1/posición del chunk esperado |
| Calibración | Score del chunk esperado vs top-1 de preguntas sin respuesta; margen mínimo de 0.05 para sugerir umbral |

Un hit@k bajo indica un problema de chunking o embeddings; un hit@k alto con respuestas malas indica un problema de
prompt o de modelo.

| Corrida | hit@1 | hit@3 | MRR |
|---|---|---|---|
| Inicial | 0.82 | 1.00 | 0.90 |
| Con sinónimos y etiquetas múltiples | 0.86 | 1.00 | 0.93 |

| Hallazgo | Acción |
|---|---|
| "Cobran delivery": el término no existía en la base ("envío"); score del esperado 0.42 | Sinónimos en `envios.md` |
| Dos preguntas con dos chunks válidos y una sola etiqueta | `expected` con varios chunks |
| Preguntas sin respuesta del dominio (top-1 de 0.37-0.54) puntúan como las que sí tienen respuesta (mínimo 0.485) | La similitud mide cercanía temática, no disponibilidad de la respuesta: lo resuelve el prompt ("no tengo esa información") |
| "Cómo estará el clima en Lima mañana" puntúa 0.605 contra `envios#tiempos-de-entrega` | Los temas ajenos los filtra el router antes del RAG |
| La calibración inicial usaba el top-1 con acierto y sugería 0.49, que dejaba sin contexto a "Cobran delivery" | Calibración con el score del chunk esperado; umbral 0.40 |

## Evaluación end-to-end

Dataset: [fase3_cases.json](../evals/fase3_cases.json). Checks adicionales por turno: `retrieves_any` (el chunk esperado
aparece en la salida de `search_policies`) y `cites_any` (la respuesta cita la fuente).

| Corrida | Casos OK |
|---|---|
| Inicial | 6/10 |
| Router con reglas ampliadas y ejemplos | 8/10 |
| Baseline de la Fase 4 (incluye la regla de descuentos) | 9/10; el caso restante se resuelve con el guard de tools (Fase 6) |

| Fallo inicial | Causa | Acción |
|---|---|---|
| Tiempo de acreditación enviado a `refunds` | El router se guiaba por la palabra "reembolso" | Regla "pregunta sobre cómo funciona algo → policies" y ejemplos few-shot |
| Hablar con una persona y seguridad enviados a `general` | Categorías incompletas en el prompt del router | Categorías ampliadas |
| Respuesta genérica pobre en `general` | `reply` vacío | `DEFAULT_GENERAL_REPLY` |
| Router de 70 s en la primera llamada | Arranque en frío de Ollama | `warm_up()` antes de medir en todos los evaluadores |
| Descuentos para estudiantes enviados a `general` tras agregar ejemplos | Regresión introducida por los ejemplos | "Cualquier pregunta sobre la tienda → policies"; ejemplo nuevo que no está en los datasets |

Los ejemplos few-shot no repiten preguntas de los datasets de evaluación (evita data leakage).

## Extensiones

| Extensión | Descripción |
|---|---|
| Otro modelo de embeddings | `OLLAMA_EMBED_MODEL=bge-m3`, reindexar y comparar métricas |
| Nuevo documento | `knowledge/cambios.md` (cambio por talla) con 3 casos de retrieval |
| Chunks más grandes | Documento completo por chunk: efecto en hit@1 y tokens de entrada |
