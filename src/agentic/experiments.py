"""Fase 4: experimentos en LangSmith (datasets, target, evaluadores heuristicos, resumen y regresiones)."""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from agentic.domain.store import reset_store
from agentic.evalkit import REPORTS_DIR, retrieval_metrics, turn_checks
from agentic.facts import extract_facts
from agentic.graph_eval import AGENT_TO_ROUTE
from agentic.multiagent import ShopAssistChat

E2E_DATASET = "shopassist-e2e"
RETRIEVAL_DATASET = "shopassist-retrieval"
CHECK_CATEGORIES = ("route", "tools", "retrieval", "content", "refunds")
# Metricas binarias cuya caida bloquea un cambio (gate de regresion)
CRITICAL_METRICS = ("checks_pass", "refunds_ok", "refund_process", "policy_compliance", "groundedness")


# ---------- datasets ----------

def case_to_example(case: dict, suite: str) -> dict:
    """Caso local (evals/*.json) -> ejemplo de LangSmith. case_id estable = clave de sincronizacion."""
    return {
        "inputs": {"turns": [t["user"] for t in case["turns"]]},
        "outputs": {"turn_specs": case["turns"], "refunds_allowed": case["refunds"], "description": case["description"]},
        "metadata": {"case_id": f"{suite}-{case['id']}", "suite": suite},
    }


def retrieval_to_example(case: dict) -> dict:
    return {
        "inputs": {"query": case["query"]},
        "outputs": {"expected": case["expected"]},
        "metadata": {"case_id": f"ret-{case['id']}", "scope": case.get("scope", "en-base" if case["expected"] else "")},
    }


def sync_dataset(client, name: str, description: str, examples: list[dict]) -> dict:
    """Upsert por case_id: crea, actualiza o borra ejemplos. LangSmith versiona el dataset en cada cambio,
    y los experimentos previos siguen apuntando a la version con la que corrieron."""
    dataset = client.read_dataset(dataset_name=name) if client.has_dataset(dataset_name=name) \
        else client.create_dataset(name, description=description)
    existing = {ex.metadata.get("case_id"): ex for ex in client.list_examples(dataset_id=dataset.id)}
    wanted = {ex["metadata"]["case_id"]: ex for ex in examples}

    created = [ex for cid, ex in wanted.items() if cid not in existing]
    if created:
        client.create_examples(dataset_id=dataset.id, examples=created)
    updated = 0
    for cid, ex in wanted.items():
        old = existing.get(cid)
        if not old:
            continue
        # LangSmith agrega claves propias a la metadata (p. ej. dataset_split): se comparan solo las nuestras
        old_meta = old.metadata or {}
        meta_changed = any(old_meta.get(k) != v for k, v in ex["metadata"].items())
        if old.inputs != ex["inputs"] or old.outputs != ex["outputs"] or meta_changed:
            client.update_example(old.id, inputs=ex["inputs"], outputs=ex["outputs"],
                                  metadata={**old_meta, **ex["metadata"]})
            updated += 1
    removed = [ex for cid, ex in existing.items() if cid not in wanted]
    for ex in removed:
        client.delete_example(ex.id)
    return {"dataset": name, "created": len(created), "updated": updated, "deleted": len(removed), "total": len(wanted)}


# ---------- target ----------

def make_e2e_target(chat: ShopAssistChat):
    """Ejecuta los turnos de un ejemplo en un hilo nuevo con el store reiniciado."""

    def target(inputs: dict) -> dict:
        store = reset_store()
        thread = chat.new_thread()
        turns = []
        for text in inputs["turns"]:
            r = chat.send(thread, text, tags=["fase4-experiment"])
            turns.append({
                "user": text,
                "answer": r.answer,
                "route": AGENT_TO_ROUTE.get(r.route, r.route),
                "agents": r.agents,
                "tools": r.tools_called,
                "tool_outputs": [{"name": s["name"], "output": s["output"]} for s in r.flow if s["type"] == "tool"],
                "hit_limit": r.hit_limit,
                "elapsed_s": r.elapsed_s,
                "route_source": r.route_source,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "cost_usd": r.cost_usd,
                "blocked_tools": r.blocked_tools,
                "guardrail_events": r.guardrail_events,
                "error": r.error,
            })
        return {"answer": turns[-1]["answer"], "turns": turns, "refunds": [x.status for x in store.refunds.values()]}

    return target


def make_retrieval_target(kb, k: int = 3):
    def target(inputs: dict) -> dict:
        docs = kb.search(inputs["query"], k=k, score_threshold=0.0)
        return {"retrieved": [d["metadata"]["chunk_id"] for d in docs], "scores": [d["metadata"]["score"] for d in docs]}

    return target


# ---------- evaluadores heuristicos ----------

def heuristic_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Checks deterministas del spec agrupados por categoria + latencia. Un resultado por metrica."""
    if not outputs or not outputs.get("turns"):
        return {"results": [{"key": "checks_pass", "score": 0, "comment": "sin salida del target"}]}

    checks: list[tuple[str, str, bool]] = []
    for i, (spec, turn) in enumerate(zip(reference_outputs["turn_specs"], outputs["turns"]), 1):
        retrieved = " ".join(t["output"] for t in turn["tool_outputs"] if t["name"] == "search_policies")
        checks += [(cat, f"T{i} {name}", ok) for cat, name, ok in
                   turn_checks(spec, turn["answer"], turn["tools"], turn["route"], retrieved, turn["hit_limit"])]
    refunds = outputs["refunds"]
    checks.append(("refunds", f"reembolsos {refunds} en {reference_outputs['refunds_allowed']}",
                   refunds in reference_outputs["refunds_allowed"]))

    failed = [name for _, name, ok in checks if not ok]
    results = [
        {"key": "checks_pass", "score": int(not failed), "comment": "; ".join(failed) or "todos OK"},
        {"key": "checks_rate", "score": round(sum(ok for *_, ok in checks) / len(checks), 3)},
        {"key": "latency_s", "score": round(sum(t["elapsed_s"] for t in outputs["turns"]), 1)},
        {"key": "tokens_total", "score": sum(t.get("tokens_in", 0) + t.get("tokens_out", 0) for t in outputs["turns"])},
        {"key": "cost_usd", "score": round(sum(t.get("cost_usd", 0.0) for t in outputs["turns"]), 6)},
        {"key": "router_rules_rate", "score": round(
            sum(t.get("route_source") == "rules" for t in outputs["turns"]) / len(outputs["turns"]), 3)},
        {"key": "guardrail_events", "score": sum(len(t.get("guardrail_events") or []) for t in outputs["turns"]),
         "comment": "; ".join(e for t in outputs["turns"] for e in (t.get("guardrail_events") or [])) or "ninguno"},
        {"key": "degraded", "score": int(any(t.get("error") for t in outputs["turns"]))},
    ]
    for cat in CHECK_CATEGORIES:
        oks = [ok for c, _, ok in checks if c == cat]
        if oks:
            results.append({"key": f"{cat}_ok", "score": int(all(oks))})
    return {"results": results}


def refund_process_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """R2 (elegibilidad antes de crear) y R3 (no afirmar reembolsos inexistentes), verificadas por codigo."""
    if not outputs or not outputs.get("turns"):
        return {"key": "refund_process", "score": 0, "comment": "sin salida del target"}
    facts = extract_facts(outputs["turns"])
    return {"key": "refund_process", "score": int(not facts.violations),
            "comment": "; ".join(facts.violations) or "R2/R3 OK"}


def retrieval_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    expected = reference_outputs["expected"]
    top = outputs["scores"][0] if outputs["scores"] else 0.0
    if not expected:  # sin respuesta en la base: solo se registra el score (calibracion del umbral)
        return {"results": [{"key": "no_answer_top_score", "score": top}]}
    m = retrieval_metrics(expected, outputs["retrieved"])
    return {"results": [
        {"key": "hit@1", "score": int(m["hit@1"])},
        {"key": "hit@k", "score": int(m["hit@k"])},
        {"key": "reciprocal_rank", "score": m["rr"]},
    ]}


# ---------- resultados ----------

def collect_rows(results) -> list[dict]:
    """ExperimentResults -> filas planas {case_id, scores, comments, run_id}."""
    rows = []
    for r in results:
        scores, comments = {}, {}
        for er in r["evaluation_results"]["results"]:
            scores[er.key] = er.score
            if er.comment:
                comments[er.key] = er.comment
        rows.append({
            "case_id": (r["example"].metadata or {}).get("case_id", str(r["example"].id)),
            "scores": scores,
            "comments": comments,
            "run_id": str(r["run"].id),
            "outputs": r["run"].outputs,
        })
    return rows


def aggregate(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Media por caso y metrica (con repeticiones, 0 < media < 1 = comportamiento inestable)."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key, score in row["scores"].items():
            if score is not None:
                acc[row["case_id"]][key].append(float(score))
    return {cid: {k: round(sum(v) / len(v), 3) for k, v in metrics.items()} for cid, metrics in sorted(acc.items())}


def overall(per_case: dict[str, dict[str, float]]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for metrics in per_case.values():
        for k, v in metrics.items():
            acc[k].append(v)
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(acc.items())}


def flaky_cases(per_case: dict[str, dict[str, float]], metrics=CRITICAL_METRICS) -> dict[str, list[str]]:
    return {cid: [m for m in metrics if 0 < scores.get(m, 0) < 1]
            for cid, scores in per_case.items() if any(0 < scores.get(m, 0) < 1 for m in metrics)}


def save_report(name: str, meta: dict, rows: list[dict]) -> Path:
    per_case = aggregate(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"fase4_{name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **meta,
        "overall": overall(per_case),
        "per_case": per_case,
        "rows": rows,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def compare_reports(baseline: dict, candidate: dict, critical=CRITICAL_METRICS, tolerance: float = 0.0) -> dict:
    """Regresion = una metrica critica baja en un caso respecto del baseline (mas alla de la tolerancia)."""
    regressions, improvements = [], []
    for cid, cand in candidate["per_case"].items():
        base = baseline["per_case"].get(cid)
        if not base:
            continue
        for metric in critical:
            if metric in base and metric in cand:
                delta = round(cand[metric] - base[metric], 3)
                if delta < -tolerance:
                    regressions.append({"case_id": cid, "metric": metric, "baseline": base[metric], "candidate": cand[metric]})
                elif delta > tolerance:
                    improvements.append({"case_id": cid, "metric": metric, "baseline": base[metric], "candidate": cand[metric]})
    only_base = sorted(set(baseline["per_case"]) - set(candidate["per_case"]))
    only_cand = sorted(set(candidate["per_case"]) - set(baseline["per_case"]))
    return {"regressions": regressions, "improvements": improvements, "only_baseline": only_base, "only_candidate": only_cand}
