# Fase 5 — Observabilidad

## Objetivo

Operar el sistema sin depender de una sola herramienta: trazas para depurar, logs estructurados para operar,
metricas para decidir y alertar. Usarlas para optimizar el nodo mas caro (router) y validar el cambio con el gate.

## LangSmith no es toda la observabilidad

| Pilar | Pregunta | Herramienta en el proyecto |
|---|---|---|
| Trazas | ¿Que paso en *esta* conversacion, paso a paso? | LangSmith (prompts, tool calls, tokens por nodo) + timeline local (`--thread`) |
| Logs estructurados | ¿Que pasa en *todo* el sistema? Errores, auditoria | `logs/shopassist.jsonl`: un evento JSON por linea, listo para CloudWatch / Datadog / ELK |
| Metricas | ¿Como evolucionan latencia, errores y costo? ¿Cuando alertar? | [fase5_metrics.py](../scripts/fase5_metrics.py) (p50/p95, % error, costo). Prometheus/Grafana en Fase 7 (API de larga vida) |

Motivos para no depender solo de LangSmith: disponibilidad del SaaS (el 401 de Fase 4 dejo el proyecto sin trazas),
datos sensibles que no pueden salir de la infraestructura, y alertas operativas que viven en el stack de logs/metricas.
La correlacion entre ambos mundos es el `thread_id`, presente en los logs y en la metadata de LangSmith.

## Implementacion

Codigo: [observability.py](../src/agentic/observability.py)

- **`ObservabilityHandler`** (callback de LangChain): se pasa en `config["callbacks"]` de cada turno.
  LangGraph agrega `langgraph_node` a la metadata de cada llamada, por eso cada evento sabe de que nodo viene.
  Las tools del dominio devuelven `{"error": ...}` en vez de lanzar: el handler las cuenta como error igual.
- **Eventos**: `llm_call`, `llm_error`, `tool_call`, `retrieval`, `turn_end` (ver docstring del modulo).
- **Costo**: `tokens / 1M * tarifa`. La tarifa es **hipotetica** y configurable (`COST_INPUT_PER_1M`,
  `COST_OUTPUT_PER_1M`): con Ollama el costo real es CPU/RAM por hora, pero estimar "como si fuera una API" permite
  comparar disenos (p. ej. cuanto ahorra el router hibrido) y anticipar el costo de migrar a un proveedor.
- Los tests no escriben logs (`OBS_ENABLED=false` en conftest); verifican eventos con un sink en memoria.

Ejemplo de evento:
```json
{"ts": "2026-10-01T15:02:11.482+00:00", "level": "INFO", "event": "llm_call", "thread_id": "3f9a1c2b",
 "turn": 1, "node": "refunds_agent", "latency_s": 15.49, "tokens_in": 891, "tokens_out": 87, "model": "qwen3.5:4b"}
```

Consultas equivalentes en **CloudWatch Logs Insights** (si estos logs se enviaran a AWS):
```
fields node, latency_s | filter event = "llm_call" | stats pct(latency_s, 95) as p95, count(*) by node
filter event = "tool_call" and ok = 0 | stats count(*) as errores by tool
filter event = "turn_end" | stats avg(cost_usd), sum(cost_usd), pct(latency_s, 95) by bin(1h)
```

## Optimizacion guiada por datos: router hibrido

Dato de partida (Fase 3-4): el router consume ~19 s por turno con ~430 tokens de entrada, tanto como un agente,
solo para decidir a donde va el mensaje.

`ROUTER_MODE=hybrid` agrega reglas deterministas antes del LLM (`rule_route` en [multiagent.py](../src/agentic/multiagent.py)):

| Regla | Ruta |
|---|---|
| Agente activo = reembolsos y respuesta corta sin "?" ("si, confirmo", "mejor no, me lo quedo") | refunds |
| ID de pedido + intencion de devolucion ("devolver", "reembolso") | refunds |
| ID de pedido + consulta ("estado", "que compre", "cuanto pague") | orders |
| Solo saludo | general |
| Cualquier otra cosa | **LLM** (comportamiento de Fase 4) |

Simulacion offline sobre los 22 casos e2e: 17/32 turnos (53 %) resueltos por reglas, 0 en conflicto con `route_any`.
Cada decision registra su origen (`route_source`: `rules` | `llm`) en el flujo, en los logs y en el experimento
(`router_rules_rate`).

Riesgo: las reglas son fragiles ante redacciones nuevas. Mitigacion: solo casos inequivocos y fallback al LLM;
el gate de Fase 4 decide si el cambio se acepta.

## Ejecucion

```powershell
# generar trafico con logs (cualquier script del grafo escribe logs/shopassist.jsonl)
docker compose run --rm app python scripts/fase2_chat.py

# metricas y debugging local
docker compose run --rm app python scripts/fase5_metrics.py
docker compose run --rm app python scripts/fase5_metrics.py --threads
docker compose run --rm app python scripts/fase5_metrics.py --thread <thread_id>

# experimento: router hibrido vs baseline (mismo juez congelado) y gate de regresion
docker compose run --rm app python scripts/fase4_experiment.py e2e --suite f2 --router-mode hybrid --prefix hybrid
docker compose run --rm app python scripts/fase4_experiment.py e2e --suite f3 --router-mode hybrid --prefix hybrid
docker compose run --rm app python scripts/fase4_compare.py reports/fase4_e2e_baseline_<f2>.json reports/fase4_e2e_hybrid_<f2>.json
docker compose run --rm app python scripts/fase4_compare.py reports/fase4_e2e_baseline_<f3>.json reports/fase4_e2e_hybrid_<f3>.json
```

Criterio de aceptacion del router hibrido: gate sin regresiones criticas y `latency_s` menor que el baseline.

## Runbook: depurar un caso real (f2-10, reembolso creado sin confirmar)

1. **Detectar**: en el experimento baseline, `policy_compliance = 0` y `checks_pass = 0` en `f2-10`; el comentario
   del juez dice "R1: creo el reembolso en el turno 1 sin pedir confirmacion".
2. **Localizar la traza**: LangSmith → Datasets & Experiments → `shopassist-e2e` → experimento baseline → fila `f2-10`
   → abrir la traza. Localmente: `fase5_metrics.py --threads` (columna "crea reembolso") y `--thread <id>`.
3. **Aislar el paso**: en la timeline, T1 tiene `check_refund_eligibility OK` seguido de `create_refund_request OK`
   en el mismo turno. En LangSmith, abrir la segunda llamada de `refunds_agent`: su salida contiene
   *"Como ya me has dado el motivo (llego rayado), procedere a crear la solicitud"* y el tool call.
4. **Inspeccionar la entrada exacta** de esa llamada en LangSmith: system prompt con los pasos 3-5, el mensaje del
   cliente con el motivo incluido y el esquema de `create_refund_request` disponible.
5. **Hipotesis**: el modelo colapsa los pasos 3 → 4 → 5 cuando el motivo llega en el primer mensaje; la instruccion
   "pide confirmacion" compite con "responde de forma breve" y con una tool que permite actuar. Es sistematico
   (0/9 en Fase 4) y aparecio tras agregar una tool al agente (cambio del contexto, no del prompt).
6. **Decidir la correccion**: no es un problema de observabilidad ni se resuelve de forma confiable con prompt
   (ya lo intentamos en Fase 2). La accion irreversible debe estar bloqueada por codigo → **Fase 6** (gate de
   confirmacion). Criterio de exito medible: f2-05, f2-10 y f3-08 de 0/9 a 9/9 sin regresiones en el resto.

## Ejercicios

1. Correr el chat con `ROUTER_MODE=hybrid` y ver en `--thread` que turnos no llamaron al LLM del router.
2. Detener Ollama en medio de una conversacion (`docker compose stop ollama`) y buscar el `llm_error` en los logs.
3. Cambiar la tarifa a la de un proveedor real y estimar el costo mensual de 1.000 conversaciones/dia.
4. Agregar una regla nueva al router hibrido y verificar con el gate que no rompe casos.
5. Escribir la consulta de Logs Insights que alertaria si el p95 de `turn_end` supera 60 s en una hora.

## Checklist

- [ ] Explicar trazas vs logs vs metricas y que aporta cada una ademas de LangSmith.
- [ ] Explicar como se obtiene el nodo de cada llamada (callbacks + metadata de LangGraph).
- [ ] Justificar el costo estimado con tarifa hipotetica en un sistema local.
- [ ] Explicar el router hibrido: que decide por reglas, que no, y como se valida.
- [ ] Recorrer el runbook de debugging de un caso real de punta a punta.
