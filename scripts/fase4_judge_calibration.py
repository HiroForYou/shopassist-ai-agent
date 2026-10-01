"""Fase 4: calibra los evaluadores contra etiquetas humanas (evaluar al evaluador).

- groundedness: solo juez LLM.
- policy_compliance: la etiqueta humana cubre R1-R6. Se mide el juez (R1, R4-R6, con hechos verificados),
  el codigo (R2, R3: agentic.facts) y el veredicto combinado = min(juez, codigo), que es lo que se usa en experimentos.

Metricas: accuracy, FP (aprueba algo que el humano rechazo: el error peligroso) y FN (rechaza algo correcto: ruido).

Uso:
    python scripts/fase4_judge_calibration.py
    python scripts/fase4_judge_calibration.py --reasoning
    python scripts/fase4_judge_calibration.py --judge-model granite4.1:3b
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from agentic.evalkit import REPORTS_DIR, shorten, warm_up
from agentic.facts import extract_facts
from agentic.judges import JUDGES, get_judge_llm, judge

CALIBRATION_FILE = Path(__file__).resolve().parents[1] / "evals" / "judge_calibration.json"


def stats(rows: list[dict], pred_key: str, label_key: str = "label") -> str:
    rows = [r for r in rows if label_key in r]
    ok = sum(r[pred_key] == r[label_key] for r in rows)
    fp = sum(r[pred_key] == 1 and r[label_key] == 0 for r in rows)
    fn = sum(r[pred_key] == 0 and r[label_key] == 1 for r in rows)
    invalid = sum(r[pred_key] is None for r in rows)
    return f"accuracy={ok}/{len(rows)} ({ok / len(rows):.0%}) | FP={fp} | FN={fn} | invalidos={invalid}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibracion de LLM-as-judge")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--reasoning", action="store_true", help="juez con modo thinking")
    args = parser.parse_args()

    llm = get_judge_llm(args.judge_model, True if args.reasoning else None)
    items = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    print(f"Juez: {llm.model} | reasoning={llm.reasoning} | num_predict={llm.num_predict} | items: {len(items)}")
    warm_up(llm)

    rows = []
    print(f"\n{'id':<26}{'evaluador':<19}{'humano':<8}{'juez':<6}{'codigo':<8}{'final':<7}{'t':<7}comentario")
    for item in items:
        prompt, with_facts = JUDGES[item["judge"]]
        start = time.perf_counter()
        verdict = judge(llm, prompt, item["turns"], with_facts)
        elapsed = time.perf_counter() - start
        judge_score = verdict.score if verdict else None

        code_score, code_issues = None, []
        if item["judge"] == "policy_compliance":
            facts = extract_facts(item["turns"])
            code_score, code_issues = int(not facts.violations), facts.violations
        final = judge_score if code_score is None else (
            None if judge_score is None else min(judge_score, code_score))

        rows.append({**item, "judge_score": judge_score, "code_score": code_score, "final": final,
                     "elapsed_s": round(elapsed, 1), "reasoning": verdict.reasoning if verdict else "",
                     "issues": (verdict.issues if verdict else []) + code_issues})
        mark = "" if final == item["label"] else "  <-- DESACUERDO"
        if not mark and "label_judge" in item and judge_score != item["label_judge"]:
            mark = "  <-- juez acierta por la razon equivocada"
        comment = shorten("; ".join(rows[-1]["issues"]) or (verdict.reasoning if verdict else "JSON invalido"), 60)
        code = "-" if code_score is None else code_score
        print(f"{item['id']:<26}{item['judge']:<19}{item['label']:<8}{str(judge_score):<6}{str(code):<8}"
              f"{str(final):<7}{elapsed:<7.1f}{comment}{mark}")

    print()
    for name in JUDGES:
        group = [r for r in rows if r["judge"] == name]
        if not group:
            continue
        print(f"{name:<19} final   {stats(group, 'final')}")
        if name == "policy_compliance":
            # cada evaluador contra la etiqueta de SU alcance: detecta aciertos del final por la razon equivocada
            print(f"{'':<19} juez    {stats(group, 'judge_score', 'label_judge')}  (vs label_judge: R1, R4-R6)")
            print(f"{'':<19} codigo  {stats(group, 'code_score', 'label_code')}  (vs label_code: R2, R3)")
    times = sorted(r["elapsed_s"] for r in rows)
    print(f"\nLatencia del juez: mediana={times[len(times) // 2]}s | max={times[-1]}s")

    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"fase4_judge_calibration_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({"judge_model": llm.model, "reasoning": llm.reasoning, "num_predict": llm.num_predict,
                                "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Reporte: reports/{path.name}")


if __name__ == "__main__":
    main()
