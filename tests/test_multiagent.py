"""Tests del grafo multi-agent con un LLM guionado: routing, handoffs, limite de handoffs y memoria."""

import json

from langchain_core.messages import AIMessage

from agentic.multiagent import (
    DEFAULT_GENERAL_REPLY,
    MAX_HANDOFFS_PER_TURN,
    ROUTER_PROMPT,
    REFUNDS_AGENT,
    RouteDecision,
    ShopAssistChat,
    build_graph,
)


class ScriptedLLM:
    """Una sola cola de respuestas para router (RouteDecision | None) y agentes (AIMessage)."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[list] = []

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        return StructuredScripted(self)

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self._responses.pop(0)


class StructuredScripted:
    """Imita with_structured_output(include_raw=True): None en la cola = JSON invalido."""

    def __init__(self, llm: ScriptedLLM):
        self._llm = llm

    def invoke(self, messages):
        parsed = self._llm.invoke(messages)
        raw = AIMessage(content="{}", response_metadata={"total_duration": 1_500_000_000},
                        usage_metadata={"input_tokens": 100, "output_tokens": 12, "total_tokens": 112})
        return {"raw": raw, "parsed": parsed, "parsing_error": None if parsed else ValueError("json")}


def route(r: str, reply: str = "") -> RouteDecision:
    return RouteDecision(route=r, reason="test", reply=reply)


def call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class FakeKB:
    def search(self, query: str) -> list[dict]:
        return [{"page_content": "Plazo de 30 dias.", "metadata": {"chunk_id": "reembolsos#plazo", "score": 0.8}}]


def chat_with(responses: list) -> tuple[ShopAssistChat, ScriptedLLM]:
    llm = ScriptedLLM(responses)
    return ShopAssistChat(build_graph(llm, kb=FakeKB())), llm


def test_general_route_answered_by_router():
    chat, _ = chat_with([route("general", reply="Hola, en que te ayudo?")])
    result = chat.send("t1", "hola")

    assert result.answer == "Hola, en que te ayudo?"
    assert result.route == "__end__"
    assert result.tools_called == []


def test_orders_route_uses_tool():
    chat, _ = chat_with([
        route("orders"),
        call("get_order", {"order_id": "A1003"}, "c1"),
        AIMessage(content="Tu pedido esta en camino."),
    ])
    result = chat.send("t1", "estado del A1003")

    assert result.route == "orders_agent"
    assert result.tools_called == ["get_order"]
    assert result.answer == "Tu pedido esta en camino."


def test_handoff_from_orders_to_refunds():
    chat, _ = chat_with([
        route("orders"),
        call("transfer_to_refunds_agent", {"reason": "quiere reembolso"}, "c1"),
        call("check_refund_eligibility", {"order_id": "A1001"}, "c2"),
        AIMessage(content="Es elegible. Cual es el motivo?"),
    ])
    result = chat.send("t1", "info del A1001, quiero devolverlo")

    assert result.agents == ["orders_agent", REFUNDS_AGENT]
    assert result.tools_called == ["transfer_to_refunds_agent", "check_refund_eligibility"]
    state = chat.graph.get_state({"configurable": {"thread_id": "t1"}}).values
    assert state["active_agent"] == REFUNDS_AGENT


def test_policy_route_uses_rag_tool():
    chat, _ = chat_with([
        route("policies"),
        call("search_policies", {"query": "plazo de devolucion"}, "c1"),
        AIMessage(content="Tienes 30 dias [reembolsos#plazo]."),
    ])
    result = chat.send("t1", "cuantos dias tengo para devolver?")

    assert result.route == "policy_agent"
    assert result.tools_called == ["search_policies"]
    assert "reembolsos#plazo" in result.flow[2]["output"]


def test_handoff_from_policy_to_refunds():
    chat, _ = chat_with([
        route("policies"),
        call("transfer_to_refunds_agent", {"reason": "reembolso concreto"}, "c1"),
        AIMessage(content="Cual es el motivo?"),
    ])
    result = chat.send("t1", "quiero devolver el A1001")

    assert result.agents == ["policy_agent", REFUNDS_AGENT]


def test_handoff_limit_blocks_ping_pong():
    chat, _ = chat_with([
        route("orders"),
        call("transfer_to_refunds_agent", {"reason": "a"}, "c1"),
        call("transfer_to_orders_agent", {"reason": "b"}, "c2"),
        call("transfer_to_refunds_agent", {"reason": "c"}, "c3"),  # excede el limite
        AIMessage(content="No puedo resolverlo."),
    ])
    result = chat.send("t1", "loop")

    tool_steps = [s for s in result.flow if s["type"] == "tool"]
    assert len(tool_steps) == MAX_HANDOFFS_PER_TURN + 1
    assert "error" in json.loads(tool_steps[-1]["output"])
    assert result.answer == "No puedo resolverlo."
    assert result.agents == ["orders_agent", REFUNDS_AGENT]


def test_memory_persists_within_thread():
    chat, llm = chat_with([
        route("refunds"),
        AIMessage(content="Cual es tu numero de pedido?"),
        route("refunds"),
        AIMessage(content="Gracias, reviso el A1001."),
    ])
    chat.send("t1", "Quiero devolver algo")
    chat.send("t1", "Es el A1001")

    router_input_turn2 = llm.calls[2][-1].content
    assert "Quiero devolver algo" in router_input_turn2
    agent_input_turn2 = llm.calls[3]
    assert any(getattr(m, "content", "") == "Cual es tu numero de pedido?" for m in agent_input_turn2)


def test_router_metrics_are_captured():
    chat, _ = chat_with([route("general", reply="Hola")])
    step = chat.send("t1", "hola").flow[0]

    assert step["type"] == "route"
    assert step["latency_s"] == 1.5
    assert (step["tokens_in"], step["tokens_out"]) == (100, 12)


def test_router_fallback_keeps_active_agent():
    chat, _ = chat_with([
        route("refunds"),
        AIMessage(content="Cual es tu numero de pedido?"),
        None,  # JSON invalido en el segundo turno
        AIMessage(content="Reviso el A1001."),
    ])
    chat.send("t1", "Quiero devolver algo")
    result = chat.send("t1", "Es el A1001")

    assert result.route == REFUNDS_AGENT
    assert "fallback" in result.flow[0]["reason"]


def test_router_fallback_without_active_agent_goes_general():
    chat, _ = chat_with([None])
    result = chat.send("t1", "???")

    assert result.route == "__end__"
    assert result.answer == DEFAULT_GENERAL_REPLY


def test_general_without_reply_uses_default():
    chat, _ = chat_with([route("general", reply="  ")])
    assert chat.send("t1", "hmm").answer == DEFAULT_GENERAL_REPLY


def test_router_prompt_formats():
    prompt = ROUTER_PROMPT.format(active_agent="ninguno")
    assert "policies" in prompt and "ninguno" in prompt


def test_warm_up_calls_models_without_tracing():
    from agentic.evalkit import warm_up

    class Recorder:
        def __init__(self):
            self.calls = []

        def invoke(self, text):
            self.calls.append(text)

        def embed_query(self, text):
            self.calls.append(text)

    llm, emb = Recorder(), Recorder()
    warm_up(llm, emb)
    assert len(llm.calls) == 1 and len(emb.calls) == 1


def test_latency_by_node_aggregates_flow():
    from agentic.evalkit import latency_by_node

    chat, _ = chat_with([
        route("orders"),
        call("get_order", {"order_id": "A1003"}, "c1"),
        AIMessage(content="En camino.", response_metadata={"total_duration": 2_000_000_000}),
    ])
    result = chat.send("t1", "estado A1003")
    stats = latency_by_node([{"turns": [{"flow": result.flow}]}])

    assert stats["router"]["calls"] == 1
    assert stats["orders_agent"]["calls"] == 2
    assert stats["orders_agent"]["latency_total_s"] == 2.0


def test_threads_are_isolated():
    chat, llm = chat_with([
        route("general", reply="Hola"),
        route("general", reply="Hola de nuevo"),
    ])
    chat.send("a", "mensaje del hilo A")
    chat.send("b", "mensaje del hilo B")

    assert "mensaje del hilo A" not in llm.calls[1][-1].content
