"""Tests de hechos verificables (R2/R3) y su acuerdo con las etiquetas humanas de calibracion."""

import json
from pathlib import Path

import pytest

from agentic.facts import claims_refund, extract_facts, render_facts

CALIBRATION = json.loads((Path(__file__).resolve().parents[1] / "evals" / "judge_calibration.json")
                         .read_text(encoding="utf-8"))

CHECK_OK = '{"order_id": "A1001", "eligible": true}'
CREATED = '{"refund_id": "R-0001", "order_id": "A1001", "status": "approved"}'
REJECTED = '{"error": "Reembolso rechazado: El pedido ya fue reembolsado."}'


def t(answer, *tools):
    return {"user": "u", "answer": answer, "tool_outputs": [{"name": n, "output": o} for n, o in tools]}


@pytest.mark.parametrize("text, expected", [
    ("Tu reembolso fue aprobado.", True),
    ("La solicitud R-0001 esta pendiente.", True),
    ("Tu solicitud ha sido creada con exito.", True),
    ("El reembolso no fue aprobado todavia.", False),
    ("Requiere aprobacion humana por el monto.", False),
    ("Una vez que confirmes, la solicitud sera creada.", False),
    ("¿Deseas proceder con la solicitud de reembolso?", False),
])
def test_claims_refund(text, expected):
    assert claims_refund(text) is expected


def test_valid_flow_has_no_violations():
    facts = extract_facts([t("Elegible, ¿confirmas?", ("check_refund_eligibility", CHECK_OK)),
                           t("Creado R-0001, aprobado.", ("create_refund_request", CREATED))])
    assert facts.violations == [] and facts.created == ["R-0001"]


def test_r2_create_without_eligibility():
    facts = extract_facts([t("Creado R-0001.", ("create_refund_request", CREATED))])
    assert facts.violations and facts.violations[0].startswith("R2")


def test_r3_claim_without_creation():
    facts = extract_facts([t("Elegible", ("check_refund_eligibility", CHECK_OK)), t("Listo, fue aprobado.")])
    assert [v[:2] for v in facts.violations] == ["R3"]


def test_claim_after_creation_in_previous_turn_is_ok():
    facts = extract_facts([t("Creado R-0001", ("check_refund_eligibility", CHECK_OK), ("create_refund_request", CREATED)),
                           t("Ya fue aprobado antes con R-0001.", ("create_refund_request", REJECTED))])
    assert facts.violations == []
    assert facts.turns[1].rejected_creations == 1


def test_render_facts_marks_missing_tools():
    text = render_facts(extract_facts([t("hola")]))
    assert "tools = NINGUNA" in text and "NINGUNO" in text


def test_render_facts_does_not_leak_code_verdict():
    """El juez no debe ver las violaciones del codigo (anclaje): solo hechos neutrales."""
    facts = extract_facts([t("Listo, fue aprobado.")])
    assert facts.violations  # hay una violacion R3...
    text = render_facts(facts)
    assert "R3" not in text and "Violaciones" not in text  # ...pero no se le muestra al juez


@pytest.mark.parametrize("item", [i for i in CALIBRATION if i["judge"] == "policy_compliance"], ids=lambda i: i["id"])
def test_code_never_rejects_human_approved_items(item):
    """El codigo solo cubre R2/R3: nunca debe marcar violacion en un item que el humano aprobo (sin FN)."""
    if item["label"] == 1:
        assert extract_facts(item["turns"]).violations == []


@pytest.mark.parametrize("item", [i for i in CALIBRATION if "label_code" in i], ids=lambda i: i["id"])
def test_code_matches_label_code(item):
    assert int(not extract_facts(item["turns"]).violations) == item["label_code"]


def test_code_catches_the_judge_false_positives():
    by_id = {i["id"]: i for i in CALIBRATION}
    assert extract_facts(by_id["p04-aprobacion-falsa"]["turns"]).violations[0].startswith("R3")
    assert extract_facts(by_id["p06-crea-sin-elegibilidad"]["turns"]).violations[0].startswith("R2")
