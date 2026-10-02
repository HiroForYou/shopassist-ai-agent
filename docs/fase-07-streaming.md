# Fase 7: streaming y baja latencia

API HTTP con respuestas en streaming, medición de la latencia que percibe el usuario y operación con Prometheus y
Grafana.

Código: [api.py](../src/agentic/api.py) · [metrics.py](../src/agentic/metrics.py) · cliente [fase7_client.py](../scripts/fase7_client.py)

## Arquitectura

```
cliente --POST /chat/stream--> FastAPI --semáforo (1 turno a la vez)--> ShopAssistChat.iter_turn --> LangGraph
   ^                              |                                          (updates + messages)
   |<-------- SSE: meta, route, tool, guardrail, token, retract, answer, done
                                  |
                          /metrics <-- listener de eventos (Fase 5) --> Prometheus --> Grafana
```

| Endpoint | Uso |
|---|---|
| `GET /health` | Estado, modo del router, guardrails y digests de modelos |
| `POST /chat` | Turno completo en JSON |
| `POST /chat/stream` | Turno en SSE; `mode`: `guarded` u `optimistic` |
| `GET /metrics` | Métricas Prometheus |
| `POST /admin/reset` | Reinicia pedidos y reembolsos (solo local, `API_ALLOW_RESET`) |
| `GET /docs` | Swagger |

## Diseño

| Métrica de latencia | Definición |
|---|---|
| Tiempo al primer evento | Primer progreso visible (ruta, tool) |
| TTFT | Primer fragmento de texto de la respuesta |
| Tiempo a la respuesta | Respuesta completa y validada |
| Latencia total | Duración del turno (métrica de las Fases 1-6) |

En CPU un turno tarda decenas de segundos. El streaming no reduce la latencia total; reduce el tiempo hasta que el
usuario ve progreso.

El output guard (Fase 6) valida la respuesta completa y el streaming envía texto antes de que exista. Dos modos:

| Modo | Lo que recibe el cliente | Riesgo |
|---|---|---|
| `guarded` (por defecto) | Progreso en vivo y la respuesta después del output guard | TTFT igual al tiempo a la respuesta |
| `optimistic` | Tokens en vivo apenas el LLM los genera | Texto descartado después; se envía `retract` y el cliente lo borra |

| Causa de `retract` | Ejemplo |
|---|---|
| Preámbulo seguido de una tool call | "Voy a crear el reembolso" → `create_refund_request` bloqueada |
| Respuesta rechazada por el output guard | Monto en una moneda distinta de USD |

| Elemento | Implementación |
|---|---|
| Tokens desde LangGraph | `graph.stream(..., stream_mode=["updates", "messages"])`; los tokens del router (JSON de la salida estructurada) se filtran |
| Backpressure | Semáforo de capacidad 1 (Ollama en CPU); cola expuesta en `shopassist_queued_turns` |
| Métricas | Histogramas (turno, LLM por nodo, retrieval, primera salida por modo), contadores (turnos, tools, bloqueos, fallbacks, reintentos, errores, retracts) y gauges (en curso, en cola), alimentados por los eventos de la Fase 5 |
| Warm-up | `API_WARMUP=true` carga los modelos al arrancar |

## Ejecución

```powershell
docker compose build api app
docker compose up -d api
docker compose --profile monitoring up -d
docker compose run --rm app python scripts/fase7_client.py chat
docker compose run --rm app python scripts/fase7_client.py chat --mode optimistic
docker compose run --rm app python scripts/fase7_client.py bench --reps 2
curl.exe -N -X POST http://localhost:8000/chat/stream -H "Content-Type: application/json" -d "{\"message\": \"Cual es el estado del A1003?\"}"
```

| Servicio | URL |
|---|---|
| Swagger | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 (Status → Targets: `shopassist-api` UP) |
| Grafana | http://localhost:3000/d/shopassist/shopassist-api-y-agentes (acceso anónimo, solo local) |

Prometheus y Grafana suman ~300-400 MB al límite de memoria de WSL2.

El benchmark intercala los modos por escenario y alterna cuál va primero. Reporta los grupos `posicion 1` y
`posicion 2` porque la caché de prompts de Ollama acelera el segundo envío de un prompt idéntico.

## Dashboard

| Fila | Contenido | Comportamiento sin tráfico |
|---|---|---|
| Superior | Turnos acumulados, p95 de latencia del rango, bloqueos, turnos degradados y fallbacks | Muestra valores |
| Resto | Latencia por nodo, turnos por ruta, tokens por segundo, cola, errores y tools | Vacía si no hubo tráfico en los últimos 5 minutos (`rate(...[5m])`) |

| Detalle de PromQL | Tratamiento |
|---|---|
| `increase()` no cuenta el valor inicial de una serie nueva | Totales con el contador directo |
| Una serie con labels no existe hasta su primer evento | `or vector(0)` |

## Resultados (02/10, `qwen3.5:4b` en CPU, router híbrido, guardrails activos)

| Escenario | Primer progreso | Primer token (optimistic) | Respuesta completa (optimistic) | Ventaja del primer token |
|---|---|---|---|---|
| Saludo | 0.04 s | - (responde el router) | 0.05 s | - |
| Estado de pedido | 0.04 s | 35.1 s | 39.9 s | 4.8 s |
| Pregunta de política | 11.9 s (router LLM) | 38.4 s | 45.6 s | 7.2 s |
| Reembolso T1 / T2 | 0.01 s | 14.5 / 14.6 s | 19.5 / 21.0 s | 5.0 / 6.4 s |

| Hallazgo | Detalle |
|---|---|
| El progreso inmediato aporta más que el token | 8/10 turnos muestran el primer evento en 0.01-0.04 s porque decide una regla del router híbrido |
| El streaming de tokens aporta 5-7 s | La espera ocurre antes del primer token (prompt, router, tools, segunda llamada al LLM); las respuestas tienen 30-50 tokens |
| `guarded` como modo por defecto | 0 retracts y ~5 s de ventaja del modo optimistic en un dominio con dinero |
| Caché de prompts como factor de confusión | Con todos los `guarded` antes que los `optimistic`, el segundo modo salió más rápido en todo (p50 de respuesta 21 s vs 34 s); el router de la misma pregunta bajó de 22.2 s a 11.9 s. Corregido con orden intercalado y alternado |

## Extensiones

| Extensión | Descripción |
|---|---|
| Alertas en Grafana | p95 de `shopassist_turn_latency_seconds` sobre 120 s durante 5 minutos |
| Prefijos estables | Aprovechar la caché de prompts con system prompt y esquemas de tools fijos al inicio |
| Concurrencia | Varias réplicas de Ollama o GPU y `API_MAX_CONCURRENCY` mayor a 1 |
