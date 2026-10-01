# Fase 3 — RAG

## Objetivo

Responder preguntas sobre politicas de la tienda con informacion recuperada de una base de conocimiento,
citando la fuente y sin inventar cuando la respuesta no existe.

## Pipeline

```
knowledge/*.md ──chunk por "##"──> 23 chunks ──qwen3-embedding:0.6b──> Qdrant (coseno)
                                                                          │
pregunta ──instruccion + embed_query──> busqueda top-k + umbral ──────────┘──> search_policies ──> policy_agent
```

Codigo: [rag.py](../src/agentic/rag.py) · Base: [knowledge/](../knowledge/) · Agente: `policy_agent` en [multiagent.py](../src/agentic/multiagent.py)

| Componente | Decision | Motivo |
|---|---|---|
| Chunking | Una seccion `##` = un chunk (100-300 caracteres) | Las secciones ya son unidades semanticas; el id `archivo#seccion` sirve como cita. |
| Contexto del chunk | Se embebe `"<titulo> - <seccion>\n<texto>"` | Una seccion suelta pierde de que documento viene ("Plazo" de que?). |
| Embeddings | `qwen3-embedding:0.6b` via Ollama | Multilingue, ~640 MB, corre en CPU. Alternativa para comparar: `bge-m3`. |
| Instruccion en la consulta | `Instruct: ...\nQuery: <pregunta>` solo en consultas | qwen3-embedding esta entrenado asi (asimetrico); medir el efecto con `--no-instruction`. |
| Vector DB | Qdrant, distancia coseno, ids `uuid5(chunk_id)` | Ingesta idempotente: reindexar no duplica puntos. |
| Recuperacion | `top_k=3`, `score_threshold=0.40` | El umbral filtra preguntas ajenas; calibrado con la eval de retrieval. |
| Generacion | `policy_agent`: siempre busca, cita `[fuente]`, "no tengo esa informacion" si no hay | Control de alucinaciones en el prompt; se medira con LLM-as-judge en Fase 4. |
| Trazas | `KnowledgeBase.search` con `@traceable(run_type="retriever")` | LangSmith muestra los documentos recuperados y sus scores en cada traza. |

## Integracion multi-agent

- El router suma la ruta `policies`: preguntas generales sin pedido concreto.
- `orders_agent` y `refunds_agent` pueden transferir a `policy_agent` (p. ej. "¿en cuanto tiempo me devuelven el dinero?"
  en medio de un reembolso), y `policy_agent` transfiere a `refunds_agent` / `orders_agent` si hay un pedido concreto.
- "¿Se puede devolver un e-book?" → `policies` (regla general). "¿Puedo devolver el A1004?" → `refunds` (pedido concreto,
  decide `policy.py`, no el RAG). La base documenta las reglas; la decision sobre un pedido sigue siendo deterministica.

## Ejecucion

```powershell
docker compose build app                          # nueva dependencia: qdrant-client
docker compose up -d ollama qdrant
docker compose run --rm ollama-pull               # agrega qwen3-embedding:0.6b
docker compose run --rm app python scripts/fase3_ingest.py
docker compose run --rm app python scripts/fase0_healthcheck.py

# evaluacion de retrieval (rapida: solo embeddings)
docker compose run --rm app python scripts/fase3_eval.py retrieval
docker compose run --rm app python scripts/fase3_eval.py retrieval --no-instruction

# evaluacion end-to-end (sistema completo)
docker compose run --rm app python scripts/fase3_eval.py e2e

# chat (incluye policy_agent)
docker compose run --rm app python scripts/fase2_chat.py
```

Dashboard de Qdrant: http://localhost:6333/dashboard (coleccion `shop_policies`, puntos y payloads).

## Evaluacion en dos niveles

**Retrieval** ([fase3_retrieval.json](../evals/fase3_retrieval.json)): 22 preguntas parafraseadas con su(s) chunk(s)
esperado(s) y 8 sin respuesta: 5 del dominio de la tienda (`scope: dominio`) y 3 ajenas (`scope: fuera-dominio`).

| Metrica | Significado |
|---|---|
| hit@1 | El chunk correcto es el primero |
| hit@k | El chunk correcto esta entre los k recuperados (lo que ve el LLM) |
| MRR | Media de 1/posicion del chunk correcto; penaliza que aparezca abajo |
| Calibracion | Compara el score del chunk **esperado** (no el top-1) con el top-1 de preguntas sin respuesta; exige margen >= 0.05 |

Separar retrieval de generacion permite ubicar el fallo: si hit@k es bajo, el problema es chunking/embeddings;
si hit@k es alto y la respuesta es mala, el problema es el prompt o el modelo.

### Resultado de referencia y lecciones (primera corrida)

hit@1 = 0.82 · hit@3 = 1.00 · MRR = 0.90 (qwen3-embedding:0.6b, con instruccion).

| Hallazgo | Accion |
|---|---|
| hit@3 = 1.00: el LLM siempre recibe el chunk correcto | Retrieval suficiente; los fallos futuros apuntan a generacion |
| `r13` "¿Cobran delivery?": el termino no existe en la base ("envio"); score del esperado 0.42 | Sinonimos en `envios.md` (vocabulary mismatch: se corrige en el contenido) |
| `r17`, `r19`: dos chunks responden la pregunta; la etiqueta solo aceptaba uno | `expected` con varios chunks (calidad del dataset de eval) |
| Preguntas sin respuesta del dominio (0.46-0.49) puntuan como las que si tienen (min 0.42) | La similitud mide cercania tematica, no si la respuesta existe: el umbral no las detecta, lo hace el prompt |
| Preguntas ajenas (ceviche: 0.24) quedan lejos | Umbral 0.40: descarta lo ajeno sin perder chunks esperados |
| La calibracion inicial usaba el top-1 con acierto y sugeria 0.49 | Con 0.49 "¿Cobran delivery?" quedaba sin contexto; corregido para usar el score del chunk esperado |

Tras editar `knowledge/` hay que reindexar (`fase3_ingest.py`) antes de volver a evaluar.

Segunda corrida (con sinonimos y etiquetas multiples): hit@1 = 0.86 · hit@3 = 1.00 · MRR = 0.93. Hallazgo: "¿Como estara
el clima en Lima manana?" puntua 0.605 contra `envios#tiempos-de-entrega` ("Lima", "tiempo"): ni siquiera lo ajeno es
siempre separable por score. Filtrar temas ajenos es responsabilidad del router, antes del RAG (defensa en capas).

### E2E: resultado de referencia (primera corrida)

6/10 PASS. RAG correcto donde se uso: citas validas, "no tengo esa informacion" ante contexto irrelevante (caso 05).

| Fallo | Causa | Accion |
|---|---|---|
| 02 tiempo de acreditacion → `refunds` | El router se guio por la palabra "reembolso" | Regla "pregunta COMO funciona → policies" + ejemplos few-shot |
| 09 hablar con una persona, 10 seguridad → `general` | El prompt del router no listaba atencion humana ni seguridad en `policies` | Categorias ampliadas + ejemplos few-shot |
| 09/10 respuesta generica pobre | `reply` vacio en ruta `general` | `DEFAULT_GENERAL_REPLY` con la oferta de ayuda completa |
| 08 T1 crea el reembolso sin confirmar | Misma regresion que Fase 2 caso 05 | Fase 6 (gate deterministico). En T3 la re-validacion de la tool evito un reembolso duplicado |
| Router 70 s en la primera llamada | Arranque en frio de Ollama | `warm_up()` antes de medir en todos los evaluadores |

**End-to-end** ([fase3_cases.json](../evals/fase3_cases.json)): 10 casos. Checks nuevos por turno:
`retrieves_any` (el chunk esperado aparece en la salida de `search_policies`) y `cites_any` (la respuesta cita la fuente).

## Ejercicios

1. Correr `retrieval` con y sin `--no-instruction`; comparar hit@1 y MRR.
2. Subir `RAG_SCORE_THRESHOLD` a 0.49 y repetir retrieval: observar que chunks esperados se pierden.
3. Cambiar `OLLAMA_EMBED_MODEL=bge-m3`, reindexar y comparar metricas (el tamano del vector cambia: la ingesta recrea la coleccion).
4. Agregar un documento `knowledge/cambios.md` (cambio de producto por otra talla), reindexar y crear 3 casos de retrieval.
5. Probar chunks mas grandes (documento completo) y observar la caida de hit@1 y el aumento de tokens de entrada.

## Checklist

- [ ] Explicar el pipeline de ingesta y de consulta, y por que son asimetricos (instruccion solo en la consulta).
- [ ] Justificar la estrategia de chunking y el encabezado contextual.
- [ ] Explicar hit@k vs MRR y como se calibra un umbral de similitud.
- [ ] Diferenciar un fallo de retrieval de uno de generacion con datos de la eval.
- [ ] Explicar por que la elegibilidad de un pedido concreto no se decide con RAG.
