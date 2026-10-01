"""Fase 1: agente con tool calling implementado a mano (sin LangGraph).

El loop es el nucleo de cualquier framework agentico:
    LLM decide -> se ejecutan tools -> resultados vuelven al LLM -> ... -> respuesta final.
En la Fase 2 este loop se reemplaza por un grafo de LangGraph.
"""

import json
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langsmith import traceable

SYSTEM_PROMPT = """Eres ShopAssist, agente de soporte de reembolsos de una tienda online.
Reglas:
- Usa SIEMPRE las herramientas para consultar pedidos; nunca inventes datos.
- Antes de crear un reembolso, verifica la elegibilidad con check_refund_eligibility.
- Si el cliente no indica el numero de pedido, pideselo.
- Responde en espanol, de forma breve y clara."""

MAX_STEPS_MESSAGE = "No pude completar la solicitud dentro del limite de pasos."


@dataclass
class AgentResult:
    answer: str
    steps: list[dict] = field(default_factory=list)
    messages: list[BaseMessage] = field(default_factory=list)


class ToolCallingAgent:
    def __init__(self, llm, tools: list[BaseTool], system_prompt: str = SYSTEM_PROMPT, max_iterations: int = 6):
        self._tools = {t.name: t for t in tools}
        self._llm = llm.bind_tools(tools)
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations

    @traceable(name="shopassist_agent", run_type="chain")
    def run(self, user_input: str, history: list[BaseMessage] | None = None) -> AgentResult:
        messages: list[BaseMessage] = [SystemMessage(self._system_prompt), *(history or []), HumanMessage(user_input)]
        steps: list[dict] = []

        for _ in range(self._max_iterations):
            ai: AIMessage = self._llm.invoke(messages)
            messages.append(ai)
            if not ai.tool_calls:
                return AgentResult(ai.content, steps, messages)

            for call in ai.tool_calls:
                output = self._execute(call)
                steps.append({"tool": call["name"], "args": call["args"], "output": output})
                messages.append(ToolMessage(content=output, tool_call_id=call["id"], name=call["name"]))

        return AgentResult(MAX_STEPS_MESSAGE, steps, messages)

    def _execute(self, call: dict) -> str:
        tool = self._tools.get(call["name"])
        if tool is None:
            result = {"error": f"Herramienta desconocida: {call['name']}"}
        else:
            try:
                result = tool.invoke(call["args"])
            except Exception as exc:  # el error vuelve al LLM para que pueda corregirse
                result = {"error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result, ensure_ascii=False, default=str)
