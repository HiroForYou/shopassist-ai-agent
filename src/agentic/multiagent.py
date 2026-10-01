"""Fase 2-3: sistema multi-agent con LangGraph.

    START -> router --orders---> orders_agent  <-> orders_agent_tools  --+
                    --refunds--> refunds_agent <-> refunds_agent_tools --+-- handoff (transfer_to_*)
                    --policies-> policy_agent  <-> policy_agent_tools  --+
                    --general--> END (responde el router)

- Router: clasifica cada turno con salida estructurada (JSON schema via Ollama).
- Especialistas: loop ReAct propio, cada uno con sus tools. policy_agent responde con RAG (Fase 3).
- Handoff: un especialista llama transfer_to_<agente>; el edge posterior a sus tools enruta al destino.
- Memoria: checkpointer por thread_id; el historial vive en el estado del grafo.
"""

import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode
from pydantic import BaseModel, Field

from agentic.config import get_settings
from agentic.evalkit import llm_step, normalize
from agentic.observability import ObservabilityHandler, Sink, estimate_cost, log_event
from agentic.rag import KnowledgeBase, make_search_tool
from agentic.tools import check_refund_eligibility, create_refund_request, get_order

ORDERS_AGENT = "orders_agent"
REFUNDS_AGENT = "refunds_agent"
POLICY_AGENT = "policy_agent"
AGENTS = (ORDERS_AGENT, REFUNDS_AGENT, POLICY_AGENT)
ROUTE_TO_AGENT = {"orders": ORDERS_AGENT, "refunds": REFUNDS_AGENT, "policies": POLICY_AGENT}
MAX_HANDOFFS_PER_TURN = 2
RECURSION_LIMIT = 20
MAX_STEPS_MESSAGE = "No pude completar la solicitud dentro del limite de pasos."

ROUTER_PROMPT = """Eres el router de ShopAssist, soporte de una tienda online. Clasifica el ULTIMO mensaje del cliente:
- refunds: quiere devolver un producto o pedir un reembolso, o continua una conversacion de reembolso
  (da un numero de pedido, un motivo, o confirma/rechaza la solicitud).
- orders: consulta estado, producto, monto o fechas de un pedido concreto, sin pedir reembolso.
- policies: pregunta sobre como funciona la tienda, sin pedir una accion sobre un pedido concreto:
  plazos, productos no reembolsables, envios, garantia, pagos, tiempos de acreditacion de reembolsos,
  horarios, canales de contacto, hablar con una persona/agente humano, seguridad de datos o posibles fraudes.
  Si pregunta COMO funciona algo (aunque mencione la palabra reembolso) y no pide iniciar una devolucion, es policies.
  Cualquier otra pregunta sobre la tienda (productos, precios, descuentos, promociones, facturacion) tambien es
  policies, aunque no sepas si hay informacion: el agente de politicas lo verifica.
- general: SOLO saludos, agradecimientos o temas sin relacion con la tienda. Solo en este caso escribe en `reply`
  una respuesta breve en espanol y ofrece ayuda con pedidos, reembolsos o politicas.
`reason`: maximo 10 palabras.

Ejemplos:
"Quiero devolver el pedido A1001" -> refunds
"En cuanto tiempo llega el reembolso a mi tarjeta?" -> policies
"Cual es el estado del A1003?" -> orders
"Quiero hablar con una persona" -> policies
"Me pidieron mi contrasena por correo, es normal?" -> policies
"Hacen precios por mayor?" -> policies
"Como estara el clima manana?" -> general

Agente activo en la conversacion: {active_agent}."""

DEFAULT_GENERAL_REPLY = (
    "Puedo ayudarte con el estado de tus pedidos, reembolsos y las politicas de la tienda "
    "(envios, garantia, pagos y atencion). En que te ayudo?"
)

ORDERS_PROMPT = """Eres el agente de PEDIDOS de ShopAssist. Respondes consultas de estado, producto, monto y fechas.
- Usa get_order; nunca inventes datos.
- Si falta el numero de pedido, pideselo.
- Si el cliente quiere devolver un producto o pedir un reembolso, llama transfer_to_refunds_agent.
- Si pregunta por politicas generales (envios, garantia, pagos), llama transfer_to_policy_agent.
- Responde en espanol, de forma breve."""

POLICY_PROMPT = """Eres el agente de POLITICAS de ShopAssist. Respondes preguntas sobre las politicas de la tienda.
- Llama SIEMPRE search_policies antes de responder.
- Responde solo con la informacion de los resultados y cita la fuente entre corchetes, p. ej. [envios#costo-de-envio].
- Si los resultados no contienen la respuesta, di que no tienes esa informacion y ofrece escalar a un agente humano.
  No completes con conocimiento propio.
- Si el cliente quiere iniciar un reembolso de un pedido concreto, llama transfer_to_refunds_agent;
  si consulta un pedido concreto, llama transfer_to_orders_agent.
- Responde en espanol, de forma breve."""

REFUNDS_PROMPT = """Eres el agente de REEMBOLSOS de ShopAssist. Flujo obligatorio:
1. Si falta el numero de pedido, pideselo.
2. Verifica con check_refund_eligibility. Si no es elegible, explica el motivo y termina.
3. Si es elegible y el cliente no dio el motivo de la devolucion, pideselo.
4. Con el motivo, resume la solicitud (pedido y motivo; si requiere aprobacion humana, avisalo) y pide confirmacion explicita.
5. Solo cuando el cliente confirme, llama create_refund_request e informa el ID y el estado devuelto.
Reglas:
- Nunca inventes datos ni digas que un reembolso fue aprobado si la herramienta no lo confirma.
- Si el cliente solo consulta informacion de un pedido, llama transfer_to_orders_agent.
- Si pregunta por politicas generales (tiempos de acreditacion, garantia, envios), llama transfer_to_policy_agent.
- Responde en espanol, de forma breve."""


class ShopState(MessagesState):
    route: str
    route_reason: str
    active_agent: str | None
    router_step: dict  # latencia/tokens de la ultima decision del router (observabilidad)


class RouteDecision(BaseModel):
    route: Literal["orders", "refunds", "policies", "general"] = Field(description="Destino del mensaje")
    reason: str = Field(description="Justificacion de la decision, maximo 10 palabras")
    reply: str = Field(default="", description="Respuesta al cliente, solo si route=general")


# ---------- handoffs ----------

def turn_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Mensajes posteriores al ultimo mensaje del cliente (el turno en curso)."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1:]
    return messages


def count_turn_handoffs(messages: list[BaseMessage]) -> int:
    return sum(
        1 for m in turn_messages(messages)
        if isinstance(m, ToolMessage) and (m.name or "").startswith("transfer_to_") and '"transfer_to"' in m.content
    )


def make_handoff_tool(target: str, description: str):
    @tool(f"transfer_to_{target}", description=description)
    def handoff(reason: str, state: Annotated[dict, InjectedState]) -> dict:
        if count_turn_handoffs(state["messages"]) >= MAX_HANDOFFS_PER_TURN:
            return {"error": "Limite de transferencias alcanzado. Resuelve la solicitud o explica al cliente que no puedes."}
        return {"transfer_to": target, "reason": reason}

    return handoff


transfer_to_refunds = make_handoff_tool(REFUNDS_AGENT, "Transfiere la conversacion al agente de reembolsos. `reason`: por que.")
transfer_to_orders = make_handoff_tool(ORDERS_AGENT, "Transfiere la conversacion al agente de pedidos. `reason`: por que.")
transfer_to_policy = make_handoff_tool(POLICY_AGENT, "Transfiere la conversacion al agente de politicas. `reason`: por que.")


def agent_tools(kb: KnowledgeBase) -> dict[str, list]:
    return {
        ORDERS_AGENT: [get_order, transfer_to_refunds, transfer_to_policy],
        REFUNDS_AGENT: [get_order, check_refund_eligibility, create_refund_request, transfer_to_orders, transfer_to_policy],
        POLICY_AGENT: [make_search_tool(kb), transfer_to_refunds, transfer_to_orders],
    }


# ---------- nodos ----------

def transcript(messages: list[BaseMessage], last_n: int = 6) -> str:
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"Cliente: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"ShopAssist: {m.content}")
    return "\n".join(lines[-last_n:])


def fallback_decision(active_agent: str | None) -> RouteDecision:
    """Salida invalida del router: continuar con el agente activo o responder de forma generica."""
    by_agent = {agent: route for route, agent in ROUTE_TO_AGENT.items()}
    if active_agent in by_agent:
        return RouteDecision(route=by_agent[active_agent], reason="fallback: JSON invalido, sigue agente activo")
    return RouteDecision(route="general", reason="fallback: JSON invalido")


# ---------- router hibrido (Fase 5) ----------
# Solo casos inequivocos; ante cualquier duda devuelve None y decide el LLM.
ORDER_ID_RE = re.compile(r"\ba\d{4}\b")
REFUND_RE = re.compile(r"\b(devolver\w*|devuelv\w*|devolucion|reembols\w*)\b")
ORDER_INFO_RE = re.compile(r"\b(estado|donde esta|que (producto )?compre|cuanto pague)\b")
SHORT_REPLY_RE = re.compile(r"^(si|ok|dale|adelante|confirmo|procede|no|mejor no|cancela|cancelalo)\b")
GREETING_RE = re.compile(r"^(hola|buenas?|buenos dias|buenas tardes|buenas noches)([\s,.!]+(hola|buenas?|tardes|noches|dias))*[\s,.!]*$")


def rule_route(messages: list[BaseMessage], active_agent: str | None) -> RouteDecision | None:
    last = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if last is None:
        return None
    text = normalize(last.content).strip()
    words = len(text.split())
    if active_agent == REFUNDS_AGENT and words <= 6 and "?" not in text and SHORT_REPLY_RE.match(text):
        return RouteDecision(route="refunds", reason="regla: respuesta corta en flujo de reembolso")
    has_id = bool(ORDER_ID_RE.search(text))
    if has_id and REFUND_RE.search(text):
        return RouteDecision(route="refunds", reason="regla: ID de pedido + intencion de devolucion")
    if has_id and ORDER_INFO_RE.search(text):
        return RouteDecision(route="orders", reason="regla: ID de pedido + consulta de informacion")
    if GREETING_RE.match(text):
        return RouteDecision(route="general", reason="regla: saludo", reply=DEFAULT_GENERAL_REPLY)
    return None


def make_router_node(llm, mode: str = "llm") -> Callable:
    # include_raw conserva el AIMessage original: latencia y tokens del router quedan medibles
    router_llm = llm.with_structured_output(RouteDecision, include_raw=True)

    def router(state: ShopState) -> dict:
        active = state.get("active_agent")
        decision = rule_route(state["messages"], active) if mode == "hybrid" else None
        if decision is not None:
            router_step = {"latency_s": 0.0, "tokens_in": 0, "tokens_out": 0, "source": "rules"}
        else:
            out = router_llm.invoke([
                SystemMessage(ROUTER_PROMPT.format(active_agent=active or "ninguno")),
                HumanMessage(transcript(state["messages"])),
            ])
            raw = out.get("raw")
            decision = out.get("parsed") or fallback_decision(active)
            router_step = {**(llm_step(raw, "router") if isinstance(raw, AIMessage) else {}), "source": "llm"}
        target = ROUTE_TO_AGENT.get(decision.route)
        update = {
            "route": target or END,
            "route_reason": f"{decision.route}: {decision.reason}",
            "router_step": router_step,
        }
        if target:
            update["active_agent"] = target
        else:
            update["messages"] = [AIMessage(decision.reply.strip() or DEFAULT_GENERAL_REPLY, name="router")]
        return update

    return router


def make_agent_node(name: str, llm, tools: list, prompt: str) -> Callable:
    bound = llm.bind_tools(tools)

    def agent(state: ShopState) -> dict:
        ai = bound.invoke([SystemMessage(prompt), *state["messages"]])
        ai.name = name
        return {"messages": [ai], "active_agent": name}

    return agent


def after_agent(name: str) -> Callable:
    def route(state: ShopState) -> str:
        return f"{name}_tools" if state["messages"][-1].tool_calls else END

    return route


def after_tools(state: ShopState) -> str:
    """Si alguna tool del paso fue un handoff exitoso, ir al agente destino; si no, volver al agente activo."""
    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            break
        if (msg.name or "").startswith("transfer_to_"):
            try:
                target = json.loads(msg.content).get("transfer_to")
            except (json.JSONDecodeError, AttributeError):
                target = None
            if target in AGENTS:
                return target
    return state["active_agent"]


PROMPTS = {ORDERS_AGENT: ORDERS_PROMPT, REFUNDS_AGENT: REFUNDS_PROMPT, POLICY_AGENT: POLICY_PROMPT}


def prompt_version() -> str:
    """Hash corto de todos los prompts: identifica que version del sistema produjo cada experimento."""
    text = "\n".join([ROUTER_PROMPT, *(PROMPTS[name] for name in AGENTS)])
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def build_graph(llm, kb: KnowledgeBase | None = None, checkpointer=None, router_mode: str | None = None):
    tools = agent_tools(kb or KnowledgeBase())
    g = StateGraph(ShopState)
    g.add_node("router", make_router_node(llm, router_mode or get_settings().router_mode))
    for name in AGENTS:
        g.add_node(name, make_agent_node(name, llm, tools[name], PROMPTS[name]))
        g.add_node(f"{name}_tools", ToolNode(tools[name]))

    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: s["route"], [*AGENTS, END])
    for name in AGENTS:
        g.add_conditional_edges(name, after_agent(name), [f"{name}_tools", END])
        g.add_conditional_edges(f"{name}_tools", after_tools, list(AGENTS))
    return g.compile(checkpointer=checkpointer or InMemorySaver())


# ---------- ejecucion por turno ----------

@dataclass
class TurnResult:
    answer: str
    route: str
    flow: list[dict] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    hit_limit: bool = False
    route_source: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


def update_to_steps(node: str, update: dict) -> list[dict]:
    if node == "router":
        stats = update.get("router_step") or {}
        return [{
            "type": "route",
            "agent": "router",
            "route": update["route"],
            "reason": update["route_reason"],
            "latency_s": stats.get("latency_s", 0.0),
            "tokens_in": stats.get("tokens_in"),
            "tokens_out": stats.get("tokens_out"),
            "source": stats.get("source", "llm"),
        }]
    if node in AGENTS:
        return [llm_step(m, node) for m in update["messages"]]
    if node.endswith("_tools"):
        return [{"type": "tool", "agent": node.removesuffix("_tools"), "name": m.name, "output": m.content}
                for m in update["messages"]]
    return []


class ShopAssistChat:
    """Envoltorio para conversar con el grafo por thread_id y capturar el flujo de cada turno."""

    def __init__(self, graph, sink: Sink | None = None):
        self.graph = graph
        self.sink = sink or log_event  # destino de los eventos de observabilidad (Fase 5)
        self._turns: dict[str, int] = {}

    @staticmethod
    def new_thread() -> str:
        return uuid.uuid4().hex[:8]

    def send(self, thread_id: str, text: str, on_step: Callable[[dict], None] | None = None,
             tags: list[str] | None = None, metadata: dict | None = None) -> TurnResult:
        turn = self._turns[thread_id] = self._turns.get(thread_id, 0) + 1
        obs = ObservabilityHandler(thread_id, turn, self.sink)
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": RECURSION_LIMIT,
            "run_name": "shopassist_graph",
            "tags": tags or [],
            "metadata": {"thread_id": thread_id, "turn": turn, **(metadata or {})},
            "callbacks": [obs],
        }
        flow: list[dict] = []
        route, route_source = "", ""
        start = time.perf_counter()
        hit_limit = False
        try:
            for chunk in self.graph.stream({"messages": [HumanMessage(text)]}, config, stream_mode="updates"):
                for node, update in chunk.items():
                    for step in update_to_steps(node, update or {}):
                        if step["type"] == "route":
                            route, route_source = step["route"], step.get("source", "llm")
                        flow.append(step)
                        if on_step:
                            on_step(step)
        except GraphRecursionError:
            hit_limit = True

        last = self.graph.get_state(config).values["messages"][-1]
        answer = MAX_STEPS_MESSAGE if hit_limit or not isinstance(last, AIMessage) else last.content
        tools_called = [s["name"] for s in flow if s["type"] == "tool"]
        result = TurnResult(
            answer=answer,
            route=route,
            flow=flow,
            tools_called=tools_called,
            agents=list(dict.fromkeys(s["agent"] for s in flow if s["type"] == "llm")),
            elapsed_s=round(time.perf_counter() - start, 2),
            hit_limit=hit_limit,
            route_source=route_source,
            tokens_in=obs.tokens_in,
            tokens_out=obs.tokens_out,
            cost_usd=estimate_cost(obs.tokens_in, obs.tokens_out),
        )
        self.sink(
            "turn_end", thread_id=thread_id, turn=turn, route=route, route_source=route_source,
            agents=result.agents, tools=tools_called,
            handoffs=[t for t in tools_called if t.startswith("transfer_to_")],
            latency_s=result.elapsed_s, llm_calls=obs.llm_calls, tool_errors=obs.tool_errors,
            tokens_in=obs.tokens_in, tokens_out=obs.tokens_out, cost_usd=result.cost_usd, hit_limit=hit_limit,
            **{k: v for k, v in (metadata or {}).items() if k in ("case_id",)},
        )
        return result
