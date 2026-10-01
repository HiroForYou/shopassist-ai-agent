"""Almacen en memoria de pedidos y solicitudes de reembolso (simula una BD / API interna)."""

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel

_ORDERS_FILE = Path(__file__).with_name("orders.json")


class Order(BaseModel):
    order_id: str
    customer: str
    product: str
    category: str
    amount: float
    status: str
    delivered_at: date | None
    refunded: bool


class RefundRequest(BaseModel):
    refund_id: str
    order_id: str
    amount: float
    reason: str
    status: str  # approved | pending_human_approval


class OrderStore:
    def __init__(self, orders: list[Order]):
        self._orders = {o.order_id: o for o in orders}
        self.refunds: dict[str, RefundRequest] = {}

    @classmethod
    def from_file(cls, path: Path = _ORDERS_FILE) -> "OrderStore":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls([Order(**r) for r in raw])

    def get(self, order_id: str) -> Order | None:
        return self._orders.get(order_id.strip().upper())

    def add_refund(self, order: Order, reason: str, status: str) -> RefundRequest:
        refund = RefundRequest(
            refund_id=f"R-{len(self.refunds) + 1:04d}",
            order_id=order.order_id,
            amount=order.amount,
            reason=reason,
            status=status,
        )
        self.refunds[refund.refund_id] = refund
        if status == "approved":
            order.refunded = True
        return refund


_store: OrderStore | None = None


def get_store() -> OrderStore:
    global _store
    if _store is None:
        _store = OrderStore.from_file()
    return _store


def reset_store() -> OrderStore:
    global _store
    _store = OrderStore.from_file()
    return _store
