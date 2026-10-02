"""Fase 7: metricas de Prometheus para la API.

Se alimentan de los mismos eventos de observabilidad de Fase 5 (listener de `log_event`), mas metricas propias del
streaming (tiempo a la primera salida, retracciones). Buckets pensados para CPU: de 0.5 s a 5 min.
"""

from prometheus_client import Counter, Gauge, Histogram

from agentic.observability import add_listener

BUCKETS = (0.5, 1, 2, 5, 10, 20, 30, 45, 60, 90, 120, 180, 300)

TURN_LATENCY = Histogram("shopassist_turn_latency_seconds", "Latencia total del turno", ["route", "route_source"],
                         buckets=BUCKETS)
TURNS = Counter("shopassist_turns_total", "Turnos procesados", ["route"])
TURN_ERRORS = Counter("shopassist_turn_errors_total", "Turnos con respuesta degradada")
LLM_LATENCY = Histogram("shopassist_llm_call_seconds", "Latencia por llamada al LLM", ["node"], buckets=BUCKETS)
LLM_TOKENS = Counter("shopassist_llm_tokens_total", "Tokens procesados", ["node", "direction"])
LLM_FALLBACKS = Counter("shopassist_llm_fallbacks_total", "Usos del modelo de fallback")
LLM_RETRIES = Counter("shopassist_llm_retries_total", "Reintentos sobre el modelo principal")
TOOL_CALLS = Counter("shopassist_tool_calls_total", "Llamadas a tools", ["tool", "ok"])
RETRIEVAL_LATENCY = Histogram("shopassist_retrieval_seconds", "Latencia de busqueda RAG", buckets=BUCKETS)
GUARDRAIL_BLOCKS = Counter("shopassist_guardrail_blocks_total", "Bloqueos de guardrails", ["layer"])

STREAM_FIRST = Histogram("shopassist_stream_first_output_seconds",
                         "Tiempo hasta la primera salida del stream (event=progreso, token=texto, answer=respuesta)",
                         ["mode", "kind"], buckets=BUCKETS)
STREAM_RETRACTS = Counter("shopassist_stream_retracts_total", "Texto ya enviado que se retracto", ["reason"])
INFLIGHT = Gauge("shopassist_inflight_turns", "Turnos ejecutandose")
QUEUED = Gauge("shopassist_queued_turns", "Turnos esperando turno (backpressure)")


def _on_event(event: str, f: dict) -> None:
    if event == "turn_end":
        TURN_LATENCY.labels(f.get("route") or "?", f.get("route_source") or "?").observe(f.get("latency_s", 0.0))
        TURNS.labels(f.get("route") or "?").inc()
        if f.get("error"):
            TURN_ERRORS.inc()
    elif event == "llm_call":
        node = f.get("node") or "?"
        LLM_LATENCY.labels(node).observe(f.get("latency_s", 0.0))
        LLM_TOKENS.labels(node, "in").inc(f.get("tokens_in") or 0)
        LLM_TOKENS.labels(node, "out").inc(f.get("tokens_out") or 0)
    elif event == "tool_call":
        TOOL_CALLS.labels(f.get("tool") or "?", str(bool(f.get("ok"))).lower()).inc()
    elif event == "retrieval":
        RETRIEVAL_LATENCY.observe(f.get("latency_s", 0.0))
    elif event == "guardrail_block":
        GUARDRAIL_BLOCKS.labels(f.get("layer") or "?").inc()
    elif event == "llm_fallback":
        LLM_FALLBACKS.inc()
    elif event == "llm_retry":
        LLM_RETRIES.inc()


def register() -> None:
    add_listener(_on_event)
