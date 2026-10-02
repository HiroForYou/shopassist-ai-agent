"""Fase 6: guardrails deterministas (defensa en profundidad).

1. Entrada  (`sanitize_input`): enmascara tarjetas y contrasenas ANTES de que el mensaje entre al grafo
   (no llegan al LLM, a LangSmith ni a los logs) y limita el largo.
2. Tools    (`check_tool_call`, punto de control de politicas): permisos por agente y precondiciones de acciones
   irreversibles. `create_refund_request` exige:
     a) elegibilidad verificada para ESE pedido en la conversacion (check_refund_eligibility con eligible=true),
     b) que el ultimo mensaje del cliente sea una confirmacion explicita,
     c) que el asistente haya pedido confirmacion en alguna de sus ultimas 3 respuestas.
   Fases 2-5 mostraron que la regla en el prompt no se cumple de forma confiable (0/9 en f2-05, f2-10, f3-08).
3. Salida   (`check_output`): bloquea respuestas que afirman un reembolso inexistente (R3) o una verificacion
   de elegibilidad que no ocurrio.
4. Resiliencia (`ResilientChatModel`): reintento + modelo de fallback; el grafo responde degradado si todo falla.
"""

import json
import re
from collections.abc import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agentic.evalkit import normalize
from agentic.facts import claims_refund
from agentic.observability import log_event

MAX_INPUT_CHARS = 2000
BLOCK_PREFIX = "BLOQUEADO por guardrail"
TOO_LONG_REPLY = ("Tu mensaje es demasiado extenso para procesarlo. ¿Puedes resumir tu consulta en pocas lineas "
                  "e indicar el numero de pedido si aplica?")
SAFE_OUTPUT_REPLY = ("Disculpa, antes de confirmarte algo necesito verificarlo en el sistema. "
                     "¿Me indicas el numero de pedido y lo reviso?")
DEGRADED_REPLY = ("En este momento tengo problemas tecnicos para atender tu solicitud. "
                  "Intenta de nuevo en unos minutos o escribe a soporte@shopassist.example.")

# ---------- 1. entrada ----------

CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
SECRET_RE = re.compile(r"(?i)\b(contrase[nñ]a|password|clave|pin|codigo de verificacion|cvv)\b(\s*(es|:|=)\s*)(\S+)")


def sanitize_input(text: str) -> tuple[str, list[str]]:
    """Devuelve (texto enmascarado, hallazgos). Los hallazgos se registran; el valor nunca."""
    findings = []
    if CARD_RE.search(text):
        text = CARD_RE.sub("[TARJETA OCULTA]", text)
        findings.append("tarjeta")
    if SECRET_RE.search(text):
        text = SECRET_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}[SECRETO OCULTO]", text)
        findings.append("secreto")
    return text, findings


# ---------- 2. tools ----------

AFFIRMATIVE_RE = re.compile(r"^(si|ok|okay|dale|adelante|confirmo|procede|de acuerdo|correcto|perfecto|claro)\b")
NEGATIVE_RE = re.compile(r"\b(no|cancela\w*|mejor no|me lo quedo|espera)\b")


def is_explicit_confirmation(text: str) -> bool:
    """Confirmacion corta y afirmativa ("Si, confirmo", "Perfecto, si confirmo el reembolso").
    No cuenta: preguntas, negaciones, ni mensajes largos con un "confirmo" embebido (p. ej. prompt injection)."""
    norm = normalize(text).strip()
    return (len(norm.split()) <= 8 and "?" not in norm and bool(AFFIRMATIVE_RE.match(norm))
            and not NEGATIVE_RE.search(norm))


def _json(content) -> dict:
    try:
        data = json.loads(content) if isinstance(content, str) else content
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def checked_orders(messages: list[BaseMessage], eligible_only: bool = False) -> set[str]:
    """Pedidos consultados con check_refund_eligibility en la conversacion (o solo los elegibles)."""
    return {
        str(data["order_id"]).upper()
        for m in messages
        if isinstance(m, ToolMessage) and m.name == "check_refund_eligibility"
        and (data := _json(m.content)).get("order_id") and (not eligible_only or data.get("eligible") is True)
    }


def verified_orders(messages: list[BaseMessage]) -> set[str]:
    """Pedidos elegibles segun check_refund_eligibility: precondicion para crear un reembolso."""
    return checked_orders(messages, eligible_only=True)


def _last_human_index(messages: list[BaseMessage]) -> int | None:
    return next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), None)


CONFIRM_REQUEST_RE = re.compile(r"(confirm|proced|desea|deseas|reembolso|devoluci)")
CONFIRM_WINDOW = 3  # respuestas del asistente hacia atras: tolera preguntas intercaladas (p. ej. f3-08 T2 sobre PayPal)


def confirmation_requested(messages: list[BaseMessage], window: int = CONFIRM_WINDOW) -> bool:
    """Alguna de las ultimas `window` respuestas finales del asistente, ANTES del mensaje actual del cliente,
    pidio confirmacion (pregunta que menciona confirmar/proceder/reembolso). "¿Algo mas?" no cuenta."""
    idx = _last_human_index(messages)
    if idx is None:
        return False
    answers = [m for m in messages[:idx] if isinstance(m, AIMessage) and not m.tool_calls and m.content]
    return any("?" in m.content and CONFIRM_REQUEST_RE.search(normalize(m.content)) for m in answers[-window:])


def check_tool_call(agent: str, call: dict, messages: list[BaseMessage], allowed: set[str]) -> str | None:
    """None si la llamada puede ejecutarse; si no, el motivo del bloqueo (se devuelve al LLM como error)."""
    if call["name"] not in allowed:
        return f"{agent} no tiene permiso para usar {call['name']}."
    if call["name"] != "create_refund_request":
        return None
    order_id = str(call["args"].get("order_id", "")).strip().upper()
    if order_id not in verified_orders(messages):
        return (f"antes de crear un reembolso verifica la elegibilidad del pedido {order_id} con "
                "check_refund_eligibility. No afirmes que esta verificado si no lo hiciste.")
    idx = _last_human_index(messages)
    last_user = messages[idx].content if idx is not None else ""
    if not (is_explicit_confirmation(last_user) and confirmation_requested(messages)):
        return ("crear un reembolso requiere que el cliente confirme explicitamente en su ultimo mensaje, despues de "
                "que le pidas confirmacion. Resume la solicitud (pedido y motivo) y pregunta si desea proceder.")
    return None


# ---------- 3. salida ----------

VERIFICATION_RE = re.compile(r"\b(he verificado|hemos verificado|verifique|es elegible|esta elegible)\b")
UNGUARDED_OUTPUT_AGENTS = {"policy_agent"}  # describe politicas generales: "un producto danado es elegible..."
STORE_CURRENCY = "USD"
# La tienda opera solo en USD (pedidos y politicas). Fases 5-6: el agente escribio "45.9 €" y el juez no lo detecto.
FOREIGN_CURRENCY_RE = re.compile(r"€|\beuros?\b|\bsoles\b|\bs/\s?\d|\bpesos\b|\bmxn\b|\beur\b")


def check_output(agent: str, answer: str, messages: list[BaseMessage]) -> str | None:
    """None si la respuesta final es consistente con lo que hicieron las tools en la conversacion."""
    if FOREIGN_CURRENCY_RE.search(normalize(answer)):  # aplica a todos los agentes: la base tambien es solo USD
        return (f"tu respuesta usa una moneda distinta de {STORE_CURRENCY}. Todos los montos de la tienda estan en "
                f"{STORE_CURRENCY}; no inventes la moneda.")
    if agent in UNGUARDED_OUTPUT_AGENTS:
        return None
    created = any(isinstance(m, ToolMessage) and m.name == "create_refund_request" and _json(m.content).get("refund_id")
                  for m in messages)
    if claims_refund(answer) and not created:
        return ("tu respuesta afirma que un reembolso fue creado o aprobado, pero ninguna herramienta lo creo. "
                "No afirmes acciones que no ocurrieron.")
    # "es elegible" y "no es elegible" son afirmaciones de verificacion: exigen que la tool se haya consultado,
    # sin importar el resultado (replay sobre conversaciones reales: 3 falsos positivos con verified_orders)
    if VERIFICATION_RE.search(normalize(answer)) and not checked_orders(messages):
        return ("tu respuesta afirma que verificaste la elegibilidad, pero no llamaste a check_refund_eligibility. "
                "Verificala con la herramienta o no lo afirmes.")
    return None


# ---------- 4. resiliencia ----------

class ResilientChatModel:
    """Envuelve un chat model: `retries` reintentos sobre el principal y luego el fallback.
    Delega bind_tools / with_structured_output para que la resiliencia se mantenga en cada uso."""

    def __init__(self, primary, fallback=None, retries: int = 1, sink: Callable = log_event):
        self.primary, self.fallback, self.retries, self.sink = primary, fallback, retries, sink

    @property
    def model(self) -> str:
        return getattr(self.primary, "model", "?")

    @property
    def reasoning(self):
        return getattr(self.primary, "reasoning", None)

    def _wrap(self, primary, fallback) -> "ResilientChatModel":
        return ResilientChatModel(primary, fallback, self.retries, self.sink)

    def bind_tools(self, tools, **kwargs):
        return self._wrap(self.primary.bind_tools(tools, **kwargs),
                          self.fallback.bind_tools(tools, **kwargs) if self.fallback else None)

    def with_structured_output(self, schema, **kwargs):
        return self._wrap(self.primary.with_structured_output(schema, **kwargs),
                          self.fallback.with_structured_output(schema, **kwargs) if self.fallback else None)

    def invoke(self, messages, *args, **kwargs):
        error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self.primary.invoke(messages, *args, **kwargs)
            except Exception as exc:  # timeout, conexion, OOM de Ollama...
                error = exc
                self.sink("llm_retry", attempt=attempt + 1, error=f"{type(exc).__name__}: {exc}"[:300])
        if self.fallback is None:
            raise error
        self.sink("llm_fallback", model=getattr(self.fallback, "model", "?"), error=f"{type(error).__name__}"[:100])
        return self.fallback.invoke(messages, *args, **kwargs)
