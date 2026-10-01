import pytest

from agentic.tools import check_refund_eligibility, create_refund_request, get_order


@pytest.mark.parametrize(
    "order_id, eligible, approval",
    [
        ("A1001", True, False),   # dentro de plazo
        ("A1002", False, False),  # fuera de plazo
        ("A1003", False, False),  # no entregado
        ("A1004", False, False),  # digital
        ("A1005", True, True),    # monto alto
        ("A1006", False, False),  # ya reembolsado
    ],
)
def test_eligibility(order_id, eligible, approval):
    result = check_refund_eligibility.invoke({"order_id": order_id})
    assert result["eligible"] is eligible
    assert result["requires_human_approval"] is approval


def test_get_order_normalizes_id():
    assert get_order.invoke({"order_id": " a1001 "})["product"] == "Soporte de laptop"


def test_unknown_order():
    assert "error" in get_order.invoke({"order_id": "Z9999"})


def test_refund_approved_marks_order(fresh_store):
    refund = create_refund_request.invoke({"order_id": "A1001", "reason": "llego danado"})
    assert refund["status"] == "approved"
    assert fresh_store.get("A1001").refunded is True
    # un segundo intento ya no es elegible
    assert "error" in create_refund_request.invoke({"order_id": "A1001", "reason": "otra vez"})


def test_refund_high_amount_goes_to_human():
    refund = create_refund_request.invoke({"order_id": "A1005", "reason": "pixel muerto"})
    assert refund["status"] == "pending_human_approval"


def test_refund_rejected_even_if_llm_skips_check():
    assert "error" in create_refund_request.invoke({"order_id": "A1004", "reason": "no me gusto"})
