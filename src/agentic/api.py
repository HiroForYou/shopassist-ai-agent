"""Fase 7: API HTTP con streaming (SSE) y metricas de Prometheus.

Endpoints:
    GET  /health        estado y digests de modelos
    POST /chat          turno completo (JSON)
    POST /chat/stream   turno en streaming (text/event-stream)
    GET  /metrics       metricas de Prometheus
    POST /admin/reset   reinicia pedidos/reembolsos (solo dev, API_ALLOW_RESET=true)

Tension streaming vs guardrails: el output guard (Fase 6) valida la respuesta COMPLETA, pero streaming envia texto
antes de terminar. Dos modos:
    guarded     (default) progreso en vivo (route/tool/guardrail) y la respuesta recien validada. Nunca se muestra
                texto que el guard vaya a rechazar; el usuario ve actividad en segundos, el texto llega al final.
    optimistic  tokens en vivo apenas los genera el LLM; si luego se descartan (el LLM termino pidiendo una tool o el
                output guard rechazo la respuesta) se envia `retract` y el cliente borra lo mostrado.
Ollama en CPU atiende de a un turno: un semaforo serializa los turnos y expone la cola como metrica (backpressure).

Ejecutar: uvicorn agentic.api:app --host 0.0.0.0 --port 8000
"""

import json
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from agentic import metrics
from agentic.config import get_settings
from agentic.domain.store import reset_store
from agentic.evalkit import warm_up
from agentic.llm import get_resilient_chat_model, model_digests
from agentic.multiagent import ShopAssistChat, build_graph
from agentic.rag import KnowledgeBase

StreamMode = Literal["guarded", "optimistic"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)  # el limite fino (2000) lo aplica el guard de entrada
    thread_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    mode: StreamMode = "guarded"


# ---------- traduccion de eventos (pura, testeable sin servidor) ----------

def turn_events(items: Iterable[tuple], mode: StreamMode, clock=time.perf_counter) -> Iterator[tuple[str, dict]]:
    """Convierte la salida de ShopAssistChat.iter_turn en eventos SSE (nombre, datos).
    Registra los tiempos a la primera salida en el evento `done` (timing.*_s, medidos en el servidor)."""
    start = clock()
    first: dict[str, float] = {}
    streamed = False  # hay texto optimista enviado que todavia no se valido

    def mark(kind: str) -> None:
        first.setdefault(kind, round(clock() - start, 3))

    for item in items:
        kind = item[0]
        if kind == "token":
            if mode != "optimistic":
                continue
            mark("event")
            mark("token")
            streamed = True
            yield "token", {"agent": item[1], "text": item[2]}
        elif kind == "step":
            step = item[1]
            if step["type"] == "llm" and step["tool_calls"] and streamed:
                streamed = False
                yield "retract", {"reason": "tool_call"}  # el texto previo era preambulo de una tool call
            elif step["type"] == "guardrail" and streamed:
                streamed = False
                yield "retract", {"reason": f"output_guard:{step['action']}"}
            if step["type"] == "route":
                mark("event")
                yield "route", {"route": step["route"], "source": step.get("source"), "latency_s": step.get("latency_s")}
            elif step["type"] == "tool":
                mark("event")
                yield "tool", {"agent": step.get("agent"), "name": step["name"], "blocked": bool(step.get("blocked"))}
            elif step["type"] == "guardrail":
                mark("event")
                yield "guardrail", {"layer": step["layer"], "action": step["action"]}
        elif kind == "result":
            result = item[1]
            mark("event")
            mark("answer")
            yield "answer", {"text": result.answer}
            yield "done", {
                "route": result.route, "route_source": result.route_source, "agents": result.agents,
                "tools": result.tools_called, "blocked_tools": result.blocked_tools,
                "guardrail_events": result.guardrail_events, "error": result.error,
                "latency_s": result.elapsed_s, "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
                "cost_usd": result.cost_usd,
                "timing": {f"first_{k}_s": v for k, v in first.items()},
            }


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


# ---------- aplicacion ----------

def create_app(chat: ShopAssistChat | None = None, warmup: bool | None = None) -> FastAPI:
    """`chat` inyectable para tests; si es None se construye el sistema real al arrancar."""
    s = get_settings()
    metrics.register()
    gate = threading.Semaphore(max(1, s.api_max_concurrency))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if chat is None:
            kb = KnowledgeBase()
            llm = get_resilient_chat_model()
            app.state.chat = ShopAssistChat(build_graph(llm, kb=kb))
            if s.api_warmup if warmup is None else warmup:
                warm_up(llm, kb.embeddings)
        else:
            app.state.chat = chat
        yield

    app = FastAPI(title="ShopAssist API", version="0.7.0", lifespan=lifespan)

    @contextmanager
    def exclusive():
        """Serializa turnos: Ollama en CPU no gana con paralelismo; la cola queda visible en /metrics."""
        metrics.QUEUED.inc()
        gate.acquire()
        metrics.QUEUED.dec()
        metrics.INFLIGHT.inc()
        try:
            yield
        finally:  # tambien si el cliente corta la conexion a mitad del stream
            metrics.INFLIGHT.dec()
            gate.release()

    @app.get("/health")
    def health():
        st = get_settings()
        return {"status": "ok", "router_mode": st.router_mode, "guardrails": st.guardrails_enabled,
                "models": model_digests([st.ollama_model, st.ollama_fallback_model, st.ollama_embed_model])}

    @app.post("/chat")
    def chat_once(req: ChatRequest, request: Request):
        chat_: ShopAssistChat = request.app.state.chat
        thread = req.thread_id or chat_.new_thread()
        with exclusive():
            result = chat_.send(thread, req.message, tags=["api"])
        return {"thread_id": thread, "answer": result.answer, "route": result.route, "agents": result.agents,
                "tools": result.tools_called, "blocked_tools": result.blocked_tools,
                "guardrail_events": result.guardrail_events, "latency_s": result.elapsed_s, "error": result.error}

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest, request: Request):
        chat_: ShopAssistChat = request.app.state.chat
        thread = req.thread_id or chat_.new_thread()

        def body() -> Iterator[str]:
            with exclusive():
                yield sse("meta", {"thread_id": thread, "mode": req.mode})
                items = chat_.iter_turn(thread, req.message, tags=["api", f"stream-{req.mode}"],
                                        stream_tokens=req.mode == "optimistic")
                for event, data in turn_events(items, req.mode):
                    if event == "retract":
                        metrics.STREAM_RETRACTS.labels(data["reason"].split(":")[0]).inc()
                    elif event == "done":
                        for key, value in data["timing"].items():
                            kind = key.removeprefix("first_").removesuffix("_s")
                            metrics.STREAM_FIRST.labels(req.mode, kind).observe(value)
                    yield sse(event, data)

        return StreamingResponse(body(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/metrics")
    def prometheus_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/admin/reset")
    def admin_reset():
        if not get_settings().api_allow_reset:
            raise HTTPException(status_code=403, detail="reset deshabilitado (API_ALLOW_RESET=false)")
        reset_store()
        return {"status": "reset"}

    return app


app = create_app()
