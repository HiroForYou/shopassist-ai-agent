"""Fase 5: observabilidad propia (independiente de LangSmith).

- Logs estructurados: un evento JSON por linea en logs/shopassist.jsonl (formato listo para CloudWatch, Datadog, ELK).
- Callback de LangChain: mide cada llamada al LLM y a las tools (latencia, tokens, errores) con el nodo de LangGraph.
- Costo estimado por tokens (tarifa configurable: con Ollama el costo real es infraestructura, no tokens).

Eventos:
    llm_call   {thread_id, turn, node, latency_s, tokens_in, tokens_out, model}
    llm_error  {thread_id, turn, node, error}
    tool_call  {thread_id, turn, node, tool, latency_s, ok, error?}
    retrieval  {latency_s, k, results, top_score}
    turn_end   {thread_id, turn, route, route_source, agents, tools, handoffs, latency_s, tokens_in, tokens_out,
                cost_usd, hit_limit}
"""

import json
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from agentic.config import get_settings

LOG_FILE = Path(__file__).resolve().parents[2] / "logs" / "shopassist.jsonl"
_logger = logging.getLogger("shopassist")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        return json.dumps(payload, ensure_ascii=False, default=str)


def _configure() -> None:
    if _logger.handlers:
        return
    LOG_FILE.parent.mkdir(exist_ok=True)
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False  # no mezclar con la salida de consola de los scripts


_listeners: list[Callable[[str, dict], None]] = []


def add_listener(fn: Callable[[str, dict], None]) -> None:
    """Suscribe una funcion a todos los eventos (Fase 7: metricas de Prometheus). Independiente de OBS_ENABLED."""
    if fn not in _listeners:
        _listeners.append(fn)


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    for fn in _listeners:
        try:
            fn(event, fields)
        except Exception:  # una metrica rota nunca debe tumbar un turno
            pass
    if not get_settings().obs_enabled:
        return
    _configure()
    _logger.log(level, event, extra={"fields": fields})


def estimate_cost(tokens_in: int, tokens_out: int) -> float:
    s = get_settings()
    return round(tokens_in / 1e6 * s.cost_input_per_1m + tokens_out / 1e6 * s.cost_output_per_1m, 6)


Sink = Callable[..., None]  # (event, **fields)


class ObservabilityHandler(BaseCallbackHandler):
    """Callback por turno. LangGraph propaga `langgraph_node` en la metadata de cada llamada."""

    def __init__(self, thread_id: str, turn: int, sink: Sink = log_event):
        self.thread_id, self.turn, self.sink = thread_id, turn, sink
        self._starts: dict[UUID, tuple[float, str, str | None]] = {}
        self.tokens_in = self.tokens_out = 0
        self.llm_calls = self.tool_errors = 0

    def _base(self, node: str | None) -> dict:
        return {"thread_id": self.thread_id, "turn": self.turn, "node": node}

    @staticmethod
    def _node(metadata: dict | None) -> str | None:
        return (metadata or {}).get("langgraph_node")

    # ---- LLM ----
    def on_chat_model_start(self, serialized, messages, *, run_id: UUID, metadata=None, **kwargs) -> None:
        self._starts[run_id] = (time.perf_counter(), self._node(metadata) or "llm", None)

    def on_llm_end(self, response, *, run_id: UUID, **kwargs) -> None:
        start, node, _ = self._starts.pop(run_id, (time.perf_counter(), "llm", None))
        msg = getattr(response.generations[0][0], "message", None) if response.generations else None
        usage = (getattr(msg, "usage_metadata", None) or {}) if msg is not None else {}
        meta = (getattr(msg, "response_metadata", None) or {}) if msg is not None else {}
        t_in, t_out = usage.get("input_tokens", 0) or 0, usage.get("output_tokens", 0) or 0
        self.tokens_in += t_in
        self.tokens_out += t_out
        self.llm_calls += 1
        self.sink("llm_call", **self._base(node), latency_s=round(time.perf_counter() - start, 3),
                  tokens_in=t_in, tokens_out=t_out, model=meta.get("model"))

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs) -> None:
        _, node, _ = self._starts.pop(run_id, (0.0, "llm", None))
        self.sink("llm_error", **self._base(node), error=f"{type(error).__name__}: {error}")

    # ---- tools ----
    def on_tool_start(self, serialized, input_str, *, run_id: UUID, metadata=None, **kwargs) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self._starts[run_id] = (time.perf_counter(), self._node(metadata) or "tools", name)

    def on_tool_end(self, output, *, run_id: UUID, **kwargs) -> None:
        start, node, name = self._starts.pop(run_id, (time.perf_counter(), "tools", "tool"))
        content = getattr(output, "content", output)
        error = None
        try:  # las tools del dominio devuelven {"error": ...} en vez de lanzar
            data = json.loads(content) if isinstance(content, str) else content
            if isinstance(data, dict) and "error" in data:
                error = str(data["error"])
        except (json.JSONDecodeError, TypeError):
            pass
        if error:
            self.tool_errors += 1
        self.sink("tool_call", **self._base(node), tool=name, latency_s=round(time.perf_counter() - start, 3),
                  ok=error is None, **({"error": error} if error else {}))

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs) -> None:
        start, node, name = self._starts.pop(run_id, (time.perf_counter(), "tools", "tool"))
        self.tool_errors += 1
        self.sink("tool_call", **self._base(node), tool=name, latency_s=round(time.perf_counter() - start, 3),
                  ok=False, error=f"{type(error).__name__}: {error}")
