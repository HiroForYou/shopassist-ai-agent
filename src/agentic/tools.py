"""Tools expuestas al LLM. Cada una valida por su cuenta: no se confia en lo que el modelo 'cree' saber."""

from langchain_core.tools import tool

from agentic.config import get_settings
from agentic.domain.policy import evaluate_refund
from agentic.domain.store import get_store


@tool
def get_order(order_id: str) -> dict:
    """Obtiene el detalle de un pedido por su ID (formato A1234)."""
    order = get_store().get(order_id)
    if order is None:
        return {"error": f"No existe el pedido {order_id}."}
    return order.model_dump(mode="json")


@tool
def check_refund_eligibility(order_id: str) -> dict:
    """Verifica si un pedido es elegible para reembolso segun la politica de la tienda."""
    order = get_store().get(order_id)
    if order is None:
        return {"error": f"No existe el pedido {order_id}."}
    result = evaluate_refund(order, get_settings().app_today)
    return {"order_id": order.order_id, **result.__dict__}


@tool
def create_refund_request(order_id: str, reason: str) -> dict:
    """Crea una solicitud de reembolso. Usar solo despues de confirmar la elegibilidad."""
    store = get_store()
    order = store.get(order_id)
    if order is None:
        return {"error": f"No existe el pedido {order_id}."}
    # Se re-evalua la politica aunque el agente ya la haya consultado.
    result = evaluate_refund(order, get_settings().app_today)
    if not result.eligible:
        return {"error": f"Reembolso rechazado: {result.reason}"}
    status = "pending_human_approval" if result.requires_human_approval else "approved"
    return store.add_refund(order, reason, status).model_dump()


SHOP_TOOLS = [get_order, check_refund_eligibility, create_refund_request]
