"""Tests de Fase 7: traduccion de eventos de streaming y endpoints HTTP. Sin Ollama ni servidor real."""

import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from test_multiagent import FakeKB, ScriptedLLM, call, route

from agentic.api import create_app, turn_events
from agentic.guardrails import TOO_LONG_REPLY
from agentic.multiagent import ShopAssistChat, TurnResult, build_graph


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def client_with(responses: list) -> TestClient:
    chat = ShopAssistChat(build_graph(ScriptedLLM(responses), kb=FakeKB(), router_mode="llm"))
    return TestClient(create_app(chat=chat))


# ---------- traduccion pura ----------

def fake_clock(times):
    it = iter(times)
    return lambda: next(it)


RESULT = TurnResult(answer="Es elegible. ¿Confirmas?", route="refunds_agent")


def test_guarded_mode_hides_tokens_and_reports_timing():
    items = [
        ("step", {"type": "route", "route": "refunds_agent", "source": "rules", "latency_s": 0.0}),
        ("token", "refunds_agent", "Es "),
        ("step", {"type": "tool", "agent": "refunds_agent", "name": "check_refund_eligibility"}),
        ("result", RESULT),
    ]
    events = list(turn_events(items, "guarded", clock=fake_clock([0.0, 0.5, 2.0, 9.0, 9.0])))
    names = [e for e, _ in events]

    assert names == ["route", "tool", "answer", "done"]  # sin tokens
    timing = events[-1][1]["timing"]
    assert timing == {"first_event_s": 0.5, "first_answer_s": 9.0}


def test_optimistic_mode_streams_tokens_and_retracts_tool_preamble():
    items = [
        ("token", "refunds_agent", "Voy a crear "),
        ("step", {"type": "llm", "agent": "refunds_agent", "tool_calls": [{"name": "create_refund_request"}]}),
        ("step", {"type": "tool", "agent": "refunds_agent", "name": "create_refund_request", "blocked": True}),
        ("token", "refunds_agent", "¿Confirmas?"),
        ("step", {"type": "llm", "agent": "refunds_agent", "tool_calls": []}),
        ("result", RESULT),
    ]
    events = list(turn_events(items, "optimistic"))
    names = [e for e, _ in events]

    assert names == ["token", "retract", "tool", "token", "answer", "done"]
    assert events[1][1]["reason"] == "tool_call"
    assert events[2][1]["blocked"] is True
    assert "first_token_s" in events[-1][1]["timing"]


def test_optimistic_mode_retracts_on_output_guard():
    items = [
        ("token", "refunds_agent", "Tu reembolso fue aprobado"),
        ("step", {"type": "llm", "agent": "refunds_agent", "tool_calls": []}),
        ("step", {"type": "guardrail", "layer": "output", "action": "retry", "reason": "x"}),
        ("token", "refunds_agent", "Necesito verificarlo."),
        ("result", RESULT),
    ]
    events = list(turn_events(items, "optimistic"))
    names = [e for e, _ in events]
    assert names == ["token", "retract", "guardrail", "token", "answer", "done"]
    assert events[1][1]["reason"] == "output_guard:retry"


# ---------- endpoints ----------

def test_health():
    with client_with([]) as c:
        r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_chat_json():
    with client_with([route("orders"), call("get_order", {"order_id": "A1003"}, "c1"),
                      AIMessage(content="Tu pedido esta en camino.")]) as c:
        r = c.post("/chat", json={"message": "estado del A1003", "thread_id": "t1"})
    body = r.json()
    assert r.status_code == 200 and body["answer"] == "Tu pedido esta en camino."
    assert body["tools"] == ["get_order"] and body["thread_id"] == "t1"


def test_chat_stream_sse_sequence():
    with client_with([route("orders"), call("get_order", {"order_id": "A1003"}, "c1"),
                      AIMessage(content="Tu pedido esta en camino.")]) as c:
        r = c.post("/chat/stream", json={"message": "estado del A1003"})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [e for e, _ in events]
    assert names[0] == "meta" and names[-2:] == ["answer", "done"]
    assert ("tool", {"agent": "orders_agent", "name": "get_order", "blocked": False}) in events
    assert events[-2][1]["text"] == "Tu pedido esta en camino."


def test_stream_input_guard_too_long():
    with client_with([]) as c:
        r = c.post("/chat/stream", json={"message": "x" * 2500})
    events = parse_sse(r.text)
    assert events[-2] == ("answer", {"text": TOO_LONG_REPLY})


def test_invalid_request_is_rejected():
    with client_with([]) as c:
        assert c.post("/chat", json={"message": ""}).status_code == 422
        assert c.post("/chat", json={"message": "hola", "thread_id": "../../etc"}).status_code == 422
        assert c.post("/chat/stream", json={"message": "hola", "mode": "turbo"}).status_code == 422


def test_metrics_exposes_turns_after_chat():
    with client_with([route("general", reply="Hola")]) as c:
        c.post("/chat", json={"message": "hola"})
        text = c.get("/metrics").text
    assert "shopassist_turns_total" in text and "shopassist_turn_latency_seconds_bucket" in text


def test_admin_reset():
    with client_with([]) as c:
        assert c.post("/admin/reset").json() == {"status": "reset"}
