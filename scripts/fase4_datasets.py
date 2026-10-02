"""Fase 4: sincroniza los casos locales (evals/*.json) con datasets de LangSmith.

- shopassist-e2e: casos multi-turno de Fase 2 (suite f2) y Fase 3 (suite f3).
- shopassist-retrieval: preguntas de retrieval de Fase 3.
Idempotente: upsert por case_id; cada cambio genera una nueva version del dataset en LangSmith.

Uso:
    python scripts/fase4_datasets.py             # sincroniza con LangSmith
    python scripts/fase4_datasets.py --dry-run   # solo muestra que se subiria
"""

import argparse
import json
from pathlib import Path

from langsmith import Client

from agentic.experiments import E2E_DATASET, RETRIEVAL_DATASET, case_to_example, retrieval_to_example, sync_dataset

EVALS = Path(__file__).resolve().parents[1] / "evals"


def load(name: str) -> list[dict]:
    return json.loads((EVALS / name).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sincroniza evals/*.json con datasets de LangSmith")
    parser.add_argument("--dry-run", action="store_true", help="no llama a LangSmith; lista los ejemplos locales")
    args = parser.parse_args()

    e2e = [case_to_example(c, "f2") for c in load("fase2_cases.json")] + \
          [case_to_example(c, "f3") for c in load("fase3_cases.json")] + \
          [case_to_example(c, "f6") for c in load("fase6_cases.json")]
    retrieval = [retrieval_to_example(c) for c in load("fase3_retrieval.json")]
    datasets = [
        (E2E_DATASET, "Conversaciones multi-turno de ShopAssist con spec de checks (Fases 2-3)", e2e),
        (RETRIEVAL_DATASET, "Preguntas de politicas con chunks esperados (Fase 3)", retrieval),
    ]

    if args.dry_run:
        for name, _, examples in datasets:
            print(f"{name}: {len(examples)} ejemplos -> " + ", ".join(e["metadata"]["case_id"] for e in examples))
        return

    client = Client()
    for name, desc, examples in datasets:
        r = sync_dataset(client, name, desc, examples)
        print(f"{r['dataset']:<24} total={r['total']:<3} creados={r['created']:<3} "
              f"actualizados={r['updated']:<3} borrados={r['deleted']}")


if __name__ == "__main__":
    main()
