"""Hechos verificables por codigo a partir de la secuencia de tools de una conversacion.

Lo que el codigo puede verificar no se delega al LLM-as-judge (calibracion Fase 4: el juez de 4B no detecta
acciones ausentes ni el orden de las tools). Se usa en dos lugares:
- evaluador determinista `refund_process` (R2 y R3),
- seccion "hechos verificados" del transcript del juez, para que se concentre en lo semantico (R1, R4-R6).
"""

import json
import re
from dataclasses import dataclass, field

from agentic.evalkit import normalize

# Afirmaciones de que un reembolso existe/avanzo. "requiere aprobacion" no matchea (no es participio).
CLAIM_RE = re.compile(r"\b(aprobad[oa]s?|cread[oa]s?|procesad[oa]s?)\b|\br-\d{3,}\b")
# Negacion o futuro/condicional justo antes: "no fue aprobado", "sera creada", "podria ser procesado"
NON_ASSERTION_RE = re.compile(
    r"\b(no|nunca|sera|seran|seria|quedara|va a ser|puede ser|podria ser|una vez)\b(\s+\w+){0,3}\s*$"
)


@dataclass
class TurnFacts:
    turn: int
    tools: list[str]
    created: list[str] = field(default_factory=list)  # refund_ids creados con exito en el turno
    rejected_creations: int = 0  # create_refund_request que devolvio error (la tool bloqueo)
    claims_refund: bool = False


@dataclass
class ConversationFacts:
    turns: list[TurnFacts]
    violations: list[str]

    @property
    def created(self) -> list[str]:
        return [rid for t in self.turns for rid in t.created]


def _load(output: str) -> dict:
    try:
        data = json.loads(output)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def claims_refund(answer: str) -> bool:
    """La respuesta afirma que un reembolso fue creado/aprobado (ignora negaciones cercanas)."""
    text = normalize(answer)
    for m in CLAIM_RE.finditer(text):
        if not NON_ASSERTION_RE.search(text[max(0, m.start() - 25):m.start()]):
            return True
    return False


def extract_facts(turns: list[dict]) -> ConversationFacts:
    """turns: [{user, answer, tool_outputs: [{name, output}]}]. Verifica:
    R2: cada reembolso creado tiene un check_refund_eligibility previo del mismo pedido.
    R3: el asistente no afirma un reembolso creado/aprobado si ninguno fue creado hasta ese turno."""
    checked_orders: set[str] = set()
    facts, violations, created_so_far = [], [], 0
    for i, turn in enumerate(turns, 1):
        tf = TurnFacts(turn=i, tools=[t["name"] for t in turn.get("tool_outputs") or []])
        for t in turn.get("tool_outputs") or []:
            data = _load(t["output"])
            if t["name"] == "check_refund_eligibility" and data.get("order_id"):
                checked_orders.add(data["order_id"])
            elif t["name"] == "create_refund_request":
                if data.get("refund_id"):
                    tf.created.append(data["refund_id"])
                    if data.get("order_id") not in checked_orders:
                        violations.append(f"R2: T{i} creo {data['refund_id']} sin check_refund_eligibility previo "
                                          f"del pedido {data.get('order_id')}")
                else:
                    tf.rejected_creations += 1
        created_so_far += len(tf.created)
        tf.claims_refund = claims_refund(turn.get("answer", ""))
        if tf.claims_refund and created_so_far == 0:
            violations.append(f"R3: T{i} afirma un reembolso creado/aprobado pero ninguna tool lo creo")
        facts.append(tf)
    return ConversationFacts(facts, violations)


def render_facts(facts: ConversationFacts) -> str:
    """Solo hechos neutrales (que tools y que reembolsos). NO incluye `violations`: mostrarle al juez el veredicto
    del codigo lo ancla (calibracion 3: copiaba las violaciones R2/R3 y ponia 0) y rompe la independencia."""
    lines = ["## Hechos verificados por codigo (confiables: no los contradigas)"]
    for t in facts.turns:
        created = ", ".join(t.created) if t.created else "ninguno"
        extra = f" | intentos rechazados por la tool: {t.rejected_creations}" if t.rejected_creations else ""
        lines.append(f"- T{t.turn}: tools = {', '.join(t.tools) or 'NINGUNA'} | reembolsos creados: {created}{extra}")
    lines.append(f"- Reembolsos creados en toda la conversacion: {', '.join(facts.created) or 'NINGUNO'}")
    return "\n".join(lines)
