"""Tests del loop del agente con un LLM guionado: sin Ollama, deterministas y rapidos."""

import json

from langchain_core.messages import AIMessage, ToolMessage

from agentic.agent import MAX_STEPS_MESSAGE, ToolCallingAgent
from agentic.tools import SHOP_TOOLS


class ScriptedLLM:
    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.calls: list[list] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self._responses.pop(0)


def tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_tool_result_is_fed_back_to_llm():
    llm = ScriptedLLM([
        tool_call("check_refund_eligibility", {"order_id": "A1001"}),
        AIMessage(content="Tu pedido es elegible."),
    ])
    result = ToolCallingAgent(llm, SHOP_TOOLS).run("Puedo devolver A1001?")

    assert result.answer == "Tu pedido es elegible."
    assert result.steps[0]["tool"] == "check_refund_eligibility"
    tool_msg = llm.calls[1][-1]
    assert isinstance(tool_msg, ToolMessage)
    assert json.loads(tool_msg.content)["eligible"] is True


def test_unknown_tool_returns_error_to_llm():
    llm = ScriptedLLM([tool_call("delete_database", {}), AIMessage(content="No puedo hacer eso.")])
    result = ToolCallingAgent(llm, SHOP_TOOLS).run("borra todo")

    assert "error" in json.loads(result.steps[0]["output"])
    assert result.answer == "No puedo hacer eso."


def test_invalid_args_do_not_crash_agent():
    llm = ScriptedLLM([tool_call("get_order", {"wrong": "x"}), AIMessage(content="Necesito tu numero de pedido.")])
    result = ToolCallingAgent(llm, SHOP_TOOLS).run("mi pedido")

    assert "error" in json.loads(result.steps[0]["output"])


def test_stops_at_max_iterations():
    llm = ScriptedLLM([tool_call("get_order", {"order_id": "A1001"}, f"c{i}") for i in range(3)])
    result = ToolCallingAgent(llm, SHOP_TOOLS, max_iterations=3).run("loop")

    assert result.answer == MAX_STEPS_MESSAGE
    assert len(result.steps) == 3
