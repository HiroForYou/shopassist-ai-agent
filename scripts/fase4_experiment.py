"""Fase 4: corre un experimento de LangSmith sobre un dataset y guarda un reporte local para regresiones.

Uso:
    python scripts/fase4_experiment.py e2e --prefix baseline                  # heuristicos + jueces
    python scripts/fase4_experiment.py e2e --cases f2-05 f2-10 --repetitions 3 # consistencia de casos inestables
    python scripts/fase4_experiment.py e2e --suite f3 --no-judge               # rapido, solo heuristicos
    python scripts/fase4_experiment.py e2e --fallback --prefix granite         # otro modelo bajo prueba
    python scripts/fase4_experiment.py retrieval --prefix emb-qwen3
    python scripts/fase4_experiment.py e2e --local ...                         # no sube resultados a LangSmith
"""

import argparse

from langsmith import Client, evaluate

from agentic.config import get_settings
from agentic.evalkit import warm_up
from agentic.experiments import (
    CRITICAL_METRICS,
    E2E_DATASET,
    RETRIEVAL_DATASET,
    aggregate,
    collect_rows,
    flaky_cases,
    heuristic_evaluator,
    make_e2e_target,
    make_retrieval_target,
    overall,
    refund_process_evaluator,
    retrieval_evaluator,
    save_report,
)
from agentic.judges import get_judge_llm, judge_evaluators
from agentic.llm import get_chat_model, get_resilient_chat_model, model_digests
from agentic.multiagent import ShopAssistChat, build_graph, prompt_version
from agentic.rag import KnowledgeBase

TABLE_METRICS = ("checks_pass", "refund_process", "groundedness", "policy_compliance", "latency_s")


def select_examples(client: Client, dataset: str, suite: str | None, prefixes: list[str]) -> list:
    examples = list(client.list_examples(dataset_name=dataset))
    if suite:
        examples = [e for e in examples if e.metadata.get("suite") == suite]
    if prefixes:
        examples = [e for e in examples if any(e.metadata.get("case_id", "").startswith(p) for p in prefixes)]
    if not examples:
        raise SystemExit("Ningun ejemplo coincide (¿corriste scripts/fase4_datasets.py?).")
    return sorted(examples, key=lambda e: e.metadata.get("case_id", ""))


def print_summary(per_case: dict, metrics: tuple[str, ...]) -> None:
    present = [m for m in metrics if any(m in s for s in per_case.values())]
    print(f"\n{'caso':<36}" + "".join(f"{m[:17]:<19}" for m in present))
    for cid, scores in per_case.items():
        print(f"{cid:<36}" + "".join(f"{str(scores.get(m, '-')):<19}" for m in present))
    print("\nPromedio: " + " | ".join(f"{k}={v}" for k, v in overall(per_case).items()))
    if flaky := flaky_cases(per_case):
        print("Inestables (0 < media < 1 entre repeticiones): " + ", ".join(f"{c} {m}" for c, m in flaky.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimento LangSmith (Fase 4)")
    parser.add_argument("dataset", choices=["e2e", "retrieval"])
    parser.add_argument("--prefix", default="exp", help="prefijo del experimento (baseline, fix-router, ...)")
    parser.add_argument("--suite", choices=["f2", "f3", "f6"], help="e2e: solo una suite")
    parser.add_argument("--cases", nargs="*", default=[], help="prefijos de case_id (ej. f2-05 f3-08)")
    parser.add_argument("--repetitions", type=int, default=1, help="repeticiones por ejemplo (consistencia)")
    parser.add_argument("--no-judge", action="store_true", help="e2e: sin LLM-as-judge (mucho mas rapido)")
    parser.add_argument("--judge-model", default=None, help="modelo juez (por defecto JUDGE_MODEL o el del agente)")
    parser.add_argument("--judge-reasoning", action="store_true", help="juez con modo thinking")
    parser.add_argument("--fallback", action="store_true", help="agente con OLLAMA_FALLBACK_MODEL")
    parser.add_argument("--reasoning", action="store_true", help="agente con modo thinking")
    parser.add_argument("--local", action="store_true", help="no sube resultados a LangSmith (solo reporte local)")
    parser.add_argument("--router-mode", choices=["llm", "hybrid"], default=None,
                        help="e2e: router solo LLM (baseline) o hibrido reglas+LLM (Fase 5). Default: ROUTER_MODE")
    parser.add_argument("--guardrails", choices=["on", "off"], default=None,
                        help="e2e: guardrails de Fase 6 activos o no (A/B). Default: GUARDRAILS_ENABLED")
    args = parser.parse_args()

    s = get_settings()
    client = Client()
    kb = KnowledgeBase()

    if args.dataset == "retrieval":
        examples = select_examples(client, RETRIEVAL_DATASET, None, args.cases)
        target, evaluators = make_retrieval_target(kb), [retrieval_evaluator]
        meta = {"embed_model": s.ollama_embed_model, "rag_top_k": 3}
        metrics = ("hit@1", "hit@k", "reciprocal_rank", "no_answer_top_score")
    else:
        examples = select_examples(client, E2E_DATASET, args.suite, args.cases)
        overrides = {"reasoning": True} if args.reasoning else {}
        router_mode = args.router_mode or s.router_mode
        guardrails = s.guardrails_enabled if args.guardrails is None else args.guardrails == "on"
        # con guardrails el agente usa reintento + fallback; sin guardrails, el modelo directo (comportamiento previo)
        llm = (get_resilient_chat_model(**overrides) if guardrails and not args.fallback
               else get_chat_model(fallback=args.fallback, **overrides))
        chat = ShopAssistChat(build_graph(llm, kb=kb, router_mode=router_mode, guardrails=guardrails))
        target, evaluators = make_e2e_target(chat), [heuristic_evaluator, refund_process_evaluator]
        meta = {"model": llm.model, "reasoning": llm.reasoning, "embed_model": s.ollama_embed_model,
                "prompt_version": prompt_version(), "rag_threshold": s.rag_score_threshold,
                "router_mode": router_mode, "guardrails": guardrails}
        if not args.no_judge:
            judge_llm = get_judge_llm(args.judge_model, True if args.judge_reasoning else None)
            evaluators += judge_evaluators(judge_llm)
            meta |= {"judge_model": judge_llm.model, "judge_reasoning": judge_llm.reasoning}
        warm_up(llm, kb.embeddings)
        metrics = TABLE_METRICS

    used = [m for m in (meta.get("model"), meta.get("judge_model"), s.ollama_embed_model) if m]
    meta |= {"dataset": args.dataset, "repetitions": args.repetitions, "examples": len(examples),
             "model_digests": model_digests(list(dict.fromkeys(used)))}
    print("Experimento: " + " | ".join(f"{k}={v}" for k, v in meta.items()))

    results = evaluate(
        target,
        data=examples,
        evaluators=evaluators,
        experiment_prefix=f"{args.dataset}-{args.prefix}",
        metadata=meta,
        max_concurrency=0,  # secuencial: el store de pedidos es global y Ollama en CPU no gana con paralelismo
        num_repetitions=args.repetitions,
        upload_results=not args.local,
        client=client,
    )
    rows = collect_rows(results)
    per_case = aggregate(rows)
    print_summary(per_case, metrics)

    name = getattr(results, "experiment_name", f"{args.dataset}-{args.prefix}")
    path = save_report(f"{args.dataset}_{args.prefix}", {"experiment": name, **meta}, rows)
    print(f"\nExperimento LangSmith: {name}" + (" (local, no subido)" if args.local else ""))
    print(f"Reporte local: reports/{path.name}")
    if args.dataset == "e2e":
        print(f"Comparar: python scripts/fase4_compare.py <baseline.json> reports/{path.name}"
              f"  (metricas criticas: {', '.join(CRITICAL_METRICS)})")


if __name__ == "__main__":
    main()
