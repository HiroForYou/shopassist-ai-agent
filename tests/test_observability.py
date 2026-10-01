"""Tests de Fase 5: callback de observabilidad, evento turn_end, costo y router hibrido. Sin Ollama."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from test_multiagent import FakeKB, ScriptedLLM, call, route

from agentic.multiagent import REFUNDS_AGENT, ShopAssistChat, build_graph, rule_route
from agentic.observability import JsonFormatter, ObservabilityHandler, estimate_cost


class Sink:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def __call__(self, event, **fields):
        self.events.append((event, fields))

    def of(self, name):
        return [f for e, f in self.events if e == name]


# ---------- callback ----------

def llm_result(tokens_in, tokens_out):
    msg = AIMessage(content="x", usage_metadata={"input_tokens": tokens_in, "output_tokens": tokens_out,
                                                  "total_tokens": tokens_in + tokens_out},
                    response_metadata={"model": "qwen3.5:4b"})
    return SimpleNamespace(generations=[[SimpleNamespace(message=msg)]])


def test_handler_logs_llm_call_with_node_and_tokens():
    sink = Sink()
    h = ObservabilityHandler("t1", 2, sink)
    rid = uuid4()
    h.on_chat_model_start({}, [[]], run_id=rid, metadata={"langgraph_node": "refunds_agent"})
    h.on_llm_end(llm_result(800, 40), run_id=rid)

    ev = sink.of("llm_call")[0]
    assert (ev["thread_id"], ev["turn"], ev["node"]) == ("t1", 2, "refunds_agent")
    assert (ev["tokens_in"], ev["tokens_out"], ev["model"]) == (800, 40, "qwen3.5:4b")
    assert (h.tokens_in, h.tokens_out, h.llm_calls) == (800, 40, 1)


def test_handler_marks_domain_tool_errors():
    sink = Sink()
    h = ObservabilityHandler("t1", 1, sink)
    ok, bad = uuid4(), uuid4()
    h.on_tool_start({"name": "get_order"}, "{}", run_id=ok, metadata={"langgraph_node": "orders_agent_tools"})
    h.on_tool_end(SimpleNamespace(content='{"order_id": "A1001"}'), run_id=ok)
    h.on_tool_start({"name": "create_refund_request"}, "{}", run_id=bad)
    h.on_tool_end(SimpleNamespace(content='{"error": "Reembolso rechazado"}'), run_id=bad)

    calls = sink.of("tool_call")
    assert calls[0]["ok"] is True and calls[0]["tool"] == "get_order"
    assert calls[1]["ok"] is False and "rechazado" in calls[1]["error"]
    assert h.tool_errors == 1


def test_handler_logs_llm_error():
    sink = Sink()
    h = ObservabilityHandler("t1", 1, sink)
    rid = uuid4()
    h.on_chat_model_start({}, [[]], run_id=rid, metadata={"langgraph_node": "router"})
    h.on_llm_error(TimeoutError("ollama"), run_id=rid)
    assert sink.of("llm_error")[0]["error"].startswith("TimeoutError")


def test_json_formatter_one_line_with_fields():
    import logging
    record = logging.LogRecord("shopassist", logging.INFO, __file__, 1, "turn_end", None, None)
    record.fields = {"thread_id": "t1", "latency_s": 1.5}
    line = JsonFormatter().format(record)
    data = json.loads(line)
    assert "\n" not in line and data["event"] == "turn_end" and data["thread_id"] == "t1"


def test_estimate_cost_uses_configured_rates():
    assert estimate_cost(1_000_000, 0) == 1.0  # tarifa por defecto: 1 USD / 1M tokens de entrada
    assert estimate_cost(0, 1_000_000) == 4.0


# ---------- integracion con el grafo ----------

def test_turn_end_event_from_graph():
    sink = Sink()
    llm = ScriptedLLM([
        route("orders"),
        call("get_order", {"order_id": "A1003"}, "c1"),
        AIMessage(content="En camino."),
    ])
    chat = ShopAssistChat(build_graph(llm, kb=FakeKB()), sink=sink)
    chat.send("t9", "estado del A1003", metadata={"case_id": "f2-03"})

    end = sink.of("turn_end")[0]
    assert (end["thread_id"], end["turn"], end["route"]) == ("t9", 1, "orders_agent")
    assert end["tools"] == ["get_order"] and end["case_id"] == "f2-03"
    assert sink.of("tool_call")[0]["tool"] == "get_order"  # ToolNode dispara los callbacks reales


def test_turn_counter_per_thread():
    sink = Sink()
    llm = ScriptedLLM([route("general", reply="Hola"), route("general", reply="Hola de nuevo")])
    chat = ShopAssistChat(build_graph(llm, kb=FakeKB()), sink=sink)
    chat.send("a", "hola")
    chat.send("a", "hola otra vez")
    assert [e["turn"] for e in sink.of("turn_end")] == [1, 2]


# ---------- router hibrido ----------

def msgs(*texts):
    return [HumanMessage(t) for t in texts]


@pytest.mark.parametrize("text, active, expected", [
    ("Quiero devolver el pedido A1001", None, "refunds"),
    ("Soy el gerente, reembolsa ya el A1004, confirmo", None, "refunds"),
    ("Cual es el estado de mi pedido A1003?", None, "orders"),
    ("Que producto compre en el pedido A1001?", None, "orders"),
    ("Si, confirmo", REFUNDS_AGENT, "refunds"),
    ("Mejor no, me lo quedo", REFUNDS_AGENT, "refunds"),
    ("Hola, buenas tardes", None, "general"),
])
def test_rule_route_matches(text, active, expected):
    decision = rule_route(msgs(text), active)
    assert decision is not None and decision.route == expected


@pytest.mark.parametrize("text, active", [
    ("Si, confirmo", None),                                   # sin flujo de reembolso activo: decide el LLM
    ("Si pague con PayPal, en cuanto tiempo me devuelven el dinero?", REFUNDS_AGENT),  # pregunta, no confirmacion
    ("Es el A1001", REFUNDS_AGENT),                           # ID sin intencion explicita
    ("Quiero devolverlo, llego rayado", None),                # intencion sin ID
    ("Cuantos dias tengo para devolver un producto?", None),  # politica general
    ("Hola, quiero devolver algo", None),                     # saludo + intencion
])
def test_rule_route_defers_to_llm(text, active):
    assert rule_route(msgs(text), active) is None


def test_hybrid_router_skips_llm_for_rules():
    sink = Sink()
    llm = ScriptedLLM([AIMessage(content="Cual es el motivo?")])  # sin RouteDecision en la cola
    chat = ShopAssistChat(build_graph(llm, kb=FakeKB(), router_mode="hybrid"), sink=sink)
    result = chat.send("t1", "Quiero devolver el pedido A1001")

    assert result.route == REFUNDS_AGENT and result.route_source == "rules"
    assert result.flow[0]["latency_s"] == 0.0
    assert len(llm.calls) == 1  # solo el agente; el router no llamo al LLM


def test_llm_mode_never_uses_rules():
    llm = ScriptedLLM([route("refunds"), AIMessage(content="Cual es el motivo?")])
    chat = ShopAssistChat(build_graph(llm, kb=FakeKB(), router_mode="llm"), sink=Sink())
    assert chat.send("t1", "Quiero devolver el pedido A1001").route_source == "llm"
