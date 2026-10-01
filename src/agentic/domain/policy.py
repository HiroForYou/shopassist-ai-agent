"""Reglas de negocio de reembolsos. Deterministas: el LLM nunca decide la elegibilidad."""

from dataclasses import dataclass
from datetime import date

from agentic.domain.store import Order

REFUND_WINDOW_DAYS = 30
APPROVAL_THRESHOLD = 200.0
NON_REFUNDABLE_CATEGORIES = {"digital"}


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reason: str
    requires_human_approval: bool = False


def evaluate_refund(order: Order, today: date) -> Eligibility:
    if order.refunded:
        return Eligibility(False, "El pedido ya fue reembolsado.")
    if order.category in NON_REFUNDABLE_CATEGORIES:
        return Eligibility(False, "Los productos digitales no admiten reembolso.")
    if order.status != "delivered" or order.delivered_at is None:
        return Eligibility(False, "El pedido aun no fue entregado; el reembolso aplica tras la entrega.")
    days = (today - order.delivered_at).days
    if days > REFUND_WINDOW_DAYS:
        return Eligibility(False, f"Fuera de plazo: han pasado {days} dias (maximo {REFUND_WINDOW_DAYS}).")
    if order.amount > APPROVAL_THRESHOLD:
        return Eligibility(True, f"Elegible; monto mayor a {APPROVAL_THRESHOLD:.0f} USD requiere aprobacion humana.", True)
    return Eligibility(True, "Elegible para reembolso.")
