"""Fase 4: regression testing entre dos experimentos (reportes locales de fase4_experiment.py).

Sale con codigo 1 si alguna metrica critica empeora en algun caso: sirve como gate en CI.

Uso:
    python scripts/fase4_compare.py reports/fase4_e2e_baseline_X.json reports/fase4_e2e_fix_Y.json
    python scripts/fase4_compare.py --latest               # los dos reportes e2e mas recientes
    python scripts/fase4_compare.py A.json B.json --tolerance 0.34   # tolera 1 de 3 repeticiones
"""

import argparse
import json
from pathlib import Path

from agentic.evalkit import REPORTS_DIR
from agentic.experiments import compare_reports


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparacion de experimentos (regresiones)")
    parser.add_argument("baseline", nargs="?", type=Path)
    parser.add_argument("candidate", nargs="?", type=Path)
    parser.add_argument("--latest", action="store_true", help="usa los dos reportes fase4_e2e_* mas recientes")
    parser.add_argument("--tolerance", type=float, default=0.0, help="caida maxima tolerada por metrica y caso")
    args = parser.parse_args()

    if args.latest:
        reports = sorted(REPORTS_DIR.glob("fase4_e2e_*.json"), key=lambda p: p.stat().st_mtime)
        if len(reports) < 2:
            raise SystemExit("Se necesitan al menos dos reportes fase4_e2e_*.json")
        args.baseline, args.candidate = reports[-2], reports[-1]
    if not (args.baseline and args.candidate):
        parser.error("indica baseline y candidate, o --latest")

    base, cand = load(args.baseline), load(args.candidate)
    print(f"Baseline : {args.baseline.name} | {base.get('experiment')} | prompts {base.get('prompt_version')} | {base.get('model')}")
    print(f"Candidate: {args.candidate.name} | {cand.get('experiment')} | prompts {cand.get('prompt_version')} | {cand.get('model')}\n")

    keys = sorted(set(base["overall"]) | set(cand["overall"]))
    print(f"{'metrica':<22}{'baseline':<10}{'candidate':<10}delta")
    for k in keys:
        b, c = base["overall"].get(k), cand["overall"].get(k)
        delta = f"{c - b:+.3f}" if b is not None and c is not None else "-"
        print(f"{k:<22}{str(b):<10}{str(c):<10}{delta}")

    diff = compare_reports(base, cand, tolerance=args.tolerance)
    for title, items in (("REGRESIONES", diff["regressions"]), ("Mejoras", diff["improvements"])):
        print(f"\n{title} ({len(items)}):")
        for it in items:
            print(f"  {it['case_id']:<36}{it['metric']:<20}{it['baseline']} -> {it['candidate']}")
    if diff["only_baseline"] or diff["only_candidate"]:
        print(f"\nCasos no comparables: solo baseline {diff['only_baseline']} | solo candidate {diff['only_candidate']}")

    if diff["regressions"]:
        raise SystemExit(f"\nGATE: {len(diff['regressions'])} regresion(es) en metricas criticas")
    print("\nGATE: sin regresiones en metricas criticas")


if __name__ == "__main__":
    main()
