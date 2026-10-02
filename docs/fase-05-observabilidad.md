# Fase 5: observabilidad

Trazas para depurar, logs estructurados para operar y métricas para decidir, sin depender de una sola herramienta.
Con esos datos se optimiza el nodo más caro del grafo (el router) y el cambio se valida con el gate de la Fase 4.

## Pilares

| Pilar | Uso | Implementación |
|---|---|---|
| Trazas | Detalle de una conversación paso a paso | LangSmith (prompts, tool calls, tokens por nodo) y timeline local (`--thread`) |
| Logs estructurados | Eventos de todo el sistema, errores, auditoría | `logs/shopassist.jsonl`: un evento JSON por línea, compatible con CloudWatch, Datadog o ELK |
| Métricas | Evolución de latencia, errores y costo; alertas | [fase5_metrics.py](../scripts/fase5_metrics.py) y Prometheus/Grafana (Fase 7) |

| Motivo para no depender solo de LangSmith | Ejemplo |
|---|---|
| Disponibilidad del SaaS | Un error 401 de la API key dejó el proyecto sin trazas durante la Fase 4 |
| Datos sensibles | Conversaciones que no pueden salir de la infraestructura propia |
| Alertas operativas | Viven en el stack de logs y métricas |

`thread_id` correlaciona logs propios y trazas de LangSmith.

## Implementación

Código: [observability.py](../src/agentic/observability.py)

| Elemento | Detalle |
|---|---|
| `ObservabilityHandler` | Callback de LangChain pasado en `config["callbacks"]` de cada turno; LangGraph agrega `langgraph_node` a la metadata, por lo que cada evento identifica su nodo |
| Errores de tools | Las tools del dominio devuelven `{"error": ...}`; el handler los cuenta como error |
| Costo | `tokens / 1M x tarifa`, con tarifa **hipotética** configurable (`COST_INPUT_PER_1M`, `COST_OUTPUT_PER_1M`); con Ollama el costo real es infraestructura, la tarifa permite comparar diseños |
| Listeners | `add_listener` alimenta las métricas de Prometheus con los mismos eventos (Fase 7) |
| Tests | `OBS_ENABLED=false`; los eventos se verifican con un sink en memoria |

| Evento | Campos |
|---|---|
| `llm_call` | `thread_id`, `turn`, `node`, `latency_s`, `tokens_in`, `tokens_out`, `model` |
| `llm_error`, `llm_retry`, `llm_fallback` | Nodo o modelo y error |
| `tool_call` | `tool`, `latency_s`, `ok`, `error` |
| `retrieval` | `latency_s`, `k`, `results`, `top_score` |
| `guardrail_block` | `layer`, motivo (Fase 6) |
| `turn_end` | Ruta y origen, agentes, tools, bloqueos, latencia, tokens, costo, error |

```json
{"ts": "2026-10-01T15:02:11.482+00:00", "level": "INFO", "event": "llm_call", "thread_id": "3f9a1c2b",
 "turn": 1, "node": "refunds_agent", "latency_s": 15.49, "tokens_in": 891, "tokens_out": 87, "model": "qwen3.5:4b"}
```

Consultas equivalentes en CloudWatch Logs Insights:

```
fields node, latency_s | filter event = "llm_call" | stats pct(latency_s, 95) as p95, count(*) by node
filter event = "tool_call" and ok = 0 | stats count(*) as errores by tool
filter event = "turn_end" | stats avg(cost_usd), sum(cost_usd), pct(latency_s, 95) by bin(1h)
```

## Router híbrido

Dato de partida: el router consumía ~19 s por turno con ~430 tokens de entrada, igual que un agente, solo para
decidir la ruta. `ROUTER_MODE=hybrid` (valor por defecto) aplica reglas deterministas antes del LLM (`rule_route` en
[multiagent.py](../src/agentic/multiagent.py)).

| Regla | Ruta |
|---|---|
| Agente activo de reembolsos y respuesta corta sin "?" ("Sí, confirmo", "Mejor no, me lo quedo") | refunds |
| ID de pedido + intención de devolución | refunds |
| ID de pedido + consulta ("estado", "qué compré", "cuánto pagué") | orders |
| Solo saludo | general |
| Cualquier otro mensaje | LLM |

Cada decisión registra su origen (`route_source`: `rules` o `llm`) en el flujo, los logs y el experimento
(`router_rules_rate`). Simulación previa sobre los casos e2e: 17/32 turnos por reglas, 0 en conflicto con la ruta
esperada.

## Ejecución

```powershell
docker compose run --rm app python scripts/fase5_metrics.py
docker compose run --rm app python scripts/fase5_metrics.py --since 2026-10-01
docker compose run --rm app python scripts/fase5_metrics.py --threads
docker compose run --rm app python scripts/fase5_metrics.py --thread <thread_id>

# A/B del router en la misma sesión
docker compose run --rm app python scripts/fase4_experiment.py e2e --suite f2 --router-mode llm --prefix ab-llm
docker compose run --rm app python scripts/fase4_experiment.py e2e --suite f2 --router-mode hybrid --prefix ab-hybrid
docker compose run --rm app python scripts/fase4_compare.py --latest
```

## Resultados

A/B en la suite f2, misma sesión, sin contenedores ajenos en ejecución:

| Métrica | Router `llm` | Router `hybrid` | Delta |
|---|---|---|---|
| Latencia media por caso | 76.5 s | 61.5 s | **-20 %** |
| Tokens por caso | 3157 | 2554 | **-19 %** |
| Turnos resueltos por reglas | 0 % | 79 % | |
| Regresiones críticas | | 0 | Gate aprobado |

Las respuestas de los agentes fueron idénticas en ambos brazos. La variación por caso es de ±20 s; la señal fiable es
el agregado y los tokens. Decisión: `ROUTER_MODE=hybrid` por defecto.

## Hallazgos

| Hallazgo | Evidencia | Acción |
|---|---|---|
| Comparar latencias entre sesiones no es un experimento controlado | Primer intento en otro día: +37 s por caso, también en casos sin reglas; router p50 de 33 s; `docker stats` con contenedores ajenos compitiendo por CPU | A/B en la misma sesión y con la máquina sin otra carga |
| Pesos del modelo cambiantes entre días | `ollama-pull` corría con cada `docker compose run app`; el comportamiento de confirmación cambió del 30/09 al 01/10 sin cambios de código | `app` no depende de `ollama-pull`; `model_digests` en cada experimento |
| Verificación inventada | f2-05: "He verificado tu pedido" sin llamar a la tool, seguido de la creación del reembolso | Detectado por `refund_process`; bloqueado en la Fase 6 |
| Latencia de retrieval | Una búsqueda de 25.5 s y el resto de 1-9 s (embeddings 0.6B en CPU con contención) | Visible desde esta fase en el evento `retrieval` |

## Runbook de depuración

Caso de referencia: f2-10, reembolso creado sin confirmación.

| Paso | Acción | Resultado en f2-10 |
|---|---|---|
| 1. Detectar | Revisar métricas críticas del experimento | `policy_compliance = 0` y `checks_pass = 0`; el juez reporta R1 (reembolso creado en el turno 1) |
| 2. Localizar | LangSmith → experimento → fila del caso; localmente `fase5_metrics.py --threads` | Thread con la columna "crea reembolso" |
| 3. Aislar | Timeline del thread (`--thread <id>`) | T1: `check_refund_eligibility` y `create_refund_request` en el mismo turno |
| 4. Inspeccionar | Entrada exacta de la llamada en LangSmith | Prompt con los pasos de confirmación, motivo en el mensaje del cliente y esquema de `create_refund_request` disponible |
| 5. Hipótesis | Relacionar entrada y salida | El modelo colapsa los pasos de confirmación cuando el motivo llega en el primer mensaje; sistemático (0/9) |
| 6. Corregir | Elegir el mecanismo de control | Precondición en código para la acción irreversible (Fase 6); criterio: 9/9 sin regresiones |

## Extensiones

| Extensión | Descripción |
|---|---|
| Costo real | Tarifa de un proveedor real y estimación mensual para un volumen dado |
| Alertas | Logs Insights o Grafana: p95 de `turn_end` sobre 60 s en una hora |
| Nuevas reglas de router | Cada regla nueva validada con el gate |
