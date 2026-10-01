"""Fase 3: evaluacion del RAG en dos niveles.

retrieval  -> solo embeddings + Qdrant (rapido, sin LLM de chat): hit@1, hit@k, MRR y calibracion del umbral.
e2e        -> sistema multi-agent completo: routing a policy_agent, uso de search_policies, contenido y citas.

Uso:
    python scripts/fase3_eval.py retrieval
    python scripts/fase3_eval.py retrieval --k 5 --no-instruction     # A/B sin instruccion en la consulta
    python scripts/fase3_eval.py e2e
    python scripts/fase3_eval.py e2e 02 08 --quiet
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from agentic.config import get_settings
from agentic.evalkit import REPORTS_DIR, load_cases, retrieval_metrics, summarize_and_save, warm_up
from agentic.graph_eval import run_graph_case
from agentic.llm import get_chat_model
from agentic.multiagent import ShopAssistChat, build_graph
from agentic.rag import KnowledgeBase

EVALS = Path(__file__).resolve().parents[1] / "evals"
CALIBRATION_K = 10  # profundidad extra para conocer el score del chunk esperado aunque quede fuera del top-k
MIN_MARGIN = 0.05  # margen minimo entre grupos para confiar en un umbral


def print_calibration(rows: list[dict], threshold: float) -> None:
    """El umbral no debe descartar el chunk ESPERADO (no el top-1) y deberia descartar las preguntas sin respuesta."""
    in_kb = [r for r in rows if r["expected"]]
    expected_scores = [r["expected_score"] for r in in_kb if r["expected_score"] is not None]
    missing = [r["id"] for r in in_kb if r["expected_score"] is None]
    in_min = min(expected_scores)
    print(f"\nScore del chunk esperado (preguntas con respuesta): min={in_min:.3f} "
          f"mediana={sorted(expected_scores)[len(expected_scores) // 2]:.3f}"
          + (f" | fuera del top-{CALIBRATION_K}: {missing}" if missing else ""))

    for scope in ("dominio", "fuera-dominio"):
        group = [r for r in rows if r.get("scope") == scope]
        if group:
            tops = [r["top_score"] for r in group]
            label = "sin respuesta, tema de la tienda" if scope == "dominio" else "ajenas a la tienda"
            print(f"Top-1 score {label} ({len(group)}): max={max(tops):.3f} min={min(tops):.3f}")

    out_rows = [r for r in rows if not r["expected"]]
    lost = [r["id"] for r in in_kb if (r["expected_score"] or 0.0) < threshold]
    leaked = [r["id"] for r in out_rows if r["top_score"] >= threshold]
    print(f"\nUmbral actual {threshold}: chunks esperados descartados {len(lost)} {lost or ''} | "
          f"sin respuesta que pasan el filtro {len(leaked)}/{len(out_rows)} {leaked or ''}")
    print(f"Umbral maximo sin perder respuestas: ~{in_min - 0.02:.2f}")

    in_domain = [r["top_score"] for r in out_rows if r.get("scope") == "dominio"]
    off_domain = [r["top_score"] for r in out_rows if r.get("scope") == "fuera-dominio"]
    if in_domain:
        margin = in_min - max(in_domain)
        if margin >= MIN_MARGIN:
            print(f"Preguntas sin respuesta separables (margen {margin:.3f}): umbral ~{(in_min + max(in_domain)) / 2:.2f}")
        else:
            print(f"Preguntas sin respuesta del dominio NO separables por score (margen {margin:.3f} < {MIN_MARGIN}):"
                  " el umbral no las detecta; debe hacerlo el prompt (\"si no esta, dilo\").")
    if off_domain and in_min - max(off_domain) >= MIN_MARGIN:
        print(f"Preguntas ajenas separables: cualquier umbral entre {max(off_domain):.2f} y {in_min - 0.02:.2f} las descarta.")


def eval_retrieval(k: int, use_instruction: bool) -> None:
    s = get_settings()
    kb = KnowledgeBase(query_instruction=None if use_instruction else "")
    cases = json.loads((EVALS / "fase3_retrieval.json").read_text(encoding="utf-8"))
    print(f"Embeddings: {s.ollama_embed_model} | k={k} | instruccion en consulta: {use_instruction}\n")
    print(f"{'id':<5}{'rank':<6}{'top1':<7}{'esper':<7}query -> top-{k}")

    rows = []
    for case in cases:
        # sin umbral y con mas profundidad: se miden todos los scores; las metricas usan solo el top-k
        docs = kb.search(case["query"], k=max(k, CALIBRATION_K), score_threshold=0.0)
        all_ids = [d["metadata"]["chunk_id"] for d in docs]
        all_scores = [d["metadata"]["score"] for d in docs]
        ids, scores = all_ids[:k], all_scores[:k]
        expected_score = max((sc for cid, sc in zip(all_ids, all_scores) if cid in case["expected"]), default=None)
        row = {**case, "retrieved": ids, "scores": scores, "top_score": scores[0] if scores else 0.0,
               "expected_score": expected_score}
        if case["expected"]:
            row.update(retrieval_metrics(case["expected"], ids))
        rows.append(row)
        rank = row.get("rank") or ("-" if case["expected"] else "n/a")
        exp = f"{expected_score:.3f}" if expected_score is not None else ("-" if case["expected"] else "n/a")
        top = ", ".join(f"{cid.split('#')[1][:28]}({sc:.2f})" for cid, sc in zip(ids, scores))
        print(f"{case['id']:<5}{str(rank):<6}{row['top_score']:<7.3f}{exp:<7}{case['query'][:45]:<46} -> {top}")

    in_kb = [r for r in rows if r["expected"]]
    hit1 = sum(r["hit@1"] for r in in_kb) / len(in_kb)
    hitk = sum(r["hit@k"] for r in in_kb) / len(in_kb)
    mrr = sum(r["rr"] for r in in_kb) / len(in_kb)
    print(f"\nEn base ({len(in_kb)}): hit@1={hit1:.2f} | hit@{k}={hitk:.2f} | MRR={mrr:.2f}")
    print_calibration(rows, s.rag_score_threshold)

    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"fase3_retrieval_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "embed_model": s.ollama_embed_model, "k": k, "query_instruction": use_instruction,
        "summary": {"hit@1": round(hit1, 3), f"hit@{k}": round(hitk, 3), "mrr": round(mrr, 3)},
        "cases": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Reporte: reports/{path.name}")


def eval_e2e(prefixes: list[str], quiet: bool, reasoning: bool) -> None:
    cases = load_cases(EVALS / "fase3_cases.json", prefixes)
    llm = get_chat_model(**({"reasoning": True} if reasoning else {}))
    kb = KnowledgeBase()
    chat = ShopAssistChat(build_graph(llm, kb=kb))
    print(f"Modelo: {llm.model} | reasoning={llm.reasoning} | casos: {len(cases)}")
    warm_up(llm, kb.embeddings)
    start = time.perf_counter()
    results = [run_graph_case(chat, case, not quiet, "fase3-eval") for case in cases]
    meta = {"model": llm.model, "reasoning": llm.reasoning, "embed_model": get_settings().ollama_embed_model}
    summarize_and_save("fase3_e2e", meta, results, time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluacion Fase 3 (RAG)")
    parser.add_argument("mode", choices=["retrieval", "e2e"])
    parser.add_argument("cases", nargs="*", help="e2e: prefijos de id de caso")
    parser.add_argument("--k", type=int, default=3, help="retrieval: top-k")
    parser.add_argument("--no-instruction", action="store_true", help="retrieval: sin instruccion en la consulta")
    parser.add_argument("--reasoning", action="store_true", help="e2e: activa thinking")
    parser.add_argument("--quiet", action="store_true", help="e2e: solo resumen")
    args = parser.parse_args()

    if args.mode == "retrieval":
        eval_retrieval(args.k, not args.no_instruction)
    else:
        eval_e2e(args.cases, args.quiet, args.reasoning)


if __name__ == "__main__":
    main()
