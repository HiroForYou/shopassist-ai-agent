"""Utilidades compartidas por los evaluadores heuristicos de cada fase (flujo, checks, reportes)."""

import json
import unicodedata
from datetime import datetime
from pathlib import Path

import time

from langchain_core.messages import AIMessage
from langsmith import tracing_context

from agentic.config import get_settings

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def warm_up(llm, embeddings=None) -> None:
    """Carga los modelos en Ollama antes de medir: el arranque en frio no debe contaminar latencias ni p95."""
    start = time.perf_counter()
    with tracing_context(enabled=False):  # no ensucia el proyecto de LangSmith
        llm.invoke("Responde solo: OK")
        if embeddings is not None:
            embeddings.embed_query("warm up")
    print(f"Warm-up: {time.perf_counter() - start:.1f}s (excluido de las metricas)")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def shorten(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def llm_step(msg: AIMessage, agent: str | None = None) -> dict:
    """Paso de flujo a partir de una respuesta del LLM (decision, razonamiento, latencia, tokens)."""
    meta = msg.response_metadata or {}
    usage = msg.usage_metadata or {}
    return {
        "type": "llm",
        "agent": agent,
        "reasoning": msg.additional_kwargs.get("reasoning_content", ""),
        "content": msg.content,
        "tool_calls": [{"name": c["name"], "args": c["args"]} for c in msg.tool_calls],
        "latency_s": round(meta.get("total_duration", 0) / 1e9, 2),
        "tokens_in": usage.get("input_tokens"),
        "tokens_out": usage.get("output_tokens"),
    }


def print_step(step: dict) -> None:
    kind = step["type"]
    if kind == "route":
        print(f"  [ROUTER:{step.get('source', 'llm')}] {step.get('latency_s', 0)}s | tokens {step.get('tokens_in')}"
              f" -> {step.get('tokens_out')} | -> {step['route']} | {shorten(step['reason'], 200)}")
    elif kind == "tool":
        print(f"    [TOOL] {step['name']} -> {shorten(step['output'])}")
    elif kind == "llm":
        who = f"LLM {step['agent']}" if step.get("agent") else "LLM"
        print(f"  [{who}] {step['latency_s']}s | tokens {step['tokens_in']} -> {step['tokens_out']}")
        if step["reasoning"]:
            print(f"      razonamiento: {shorten(step['reasoning'], 600)}")
        if step["tool_calls"]:
            if step["content"]:
                print(f"      texto: {shorten(step['content'])}")
            for call in step["tool_calls"]:
                print(f"      decide: {call['name']}({json.dumps(call['args'], ensure_ascii=False)})")
        else:
            print("      decide: respuesta final")


def turn_checks(spec: dict, answer: str, called: list[str], route: str | None = None, retrieved_text: str = "",
                invalid_answer: bool = False) -> list[tuple[str, str, bool]]:
    """Checks deterministas de un turno como (categoria, nombre, ok).
    Categorias: content, route, tools, retrieval. Claves del spec: route_any, must_call, must_not_call,
    mention_any, mention_all, retrieves_any (en la salida de search_policies), cites_any (en la respuesta)."""
    norm = normalize(answer)
    checks = [("content", "respuesta final valida", bool(norm.strip()) and not invalid_answer)]
    if routes := spec.get("route_any"):
        checks.append(("route", f"route {route} en {routes}", route in routes))
    checks += [("tools", f"llama {t}", t in called) for t in spec.get("must_call", [])]
    checks += [("tools", f"no llama {t}", t not in called) for t in spec.get("must_not_call", [])]
    if kws := spec.get("mention_any"):
        checks.append(("content", f"menciona alguno de {kws}", any(normalize(k) in norm for k in kws)))
    if kws := spec.get("mention_all"):
        checks.append(("content", f"menciona todos {kws}", all(normalize(k) in norm for k in kws)))
    if sources := spec.get("retrieves_any"):
        checks.append(("retrieval", f"recupera alguna de {sources}", any(s in retrieved_text for s in sources)))
    if sources := spec.get("cites_any"):
        checks.append(("content", f"cita alguna de {sources}", any(s in answer for s in sources)))
    return checks


def spec_checks(spec: dict, answer: str, called: list[str], invalid_answer: bool = False) -> list[tuple[str, bool]]:
    """Version sin categorias (Fase 1: sin router ni RAG)."""
    return [(name, ok) for _, name, ok in turn_checks(spec, answer, called, invalid_answer=invalid_answer)]


def retrieval_metrics(expected: list[str], retrieved: list[str]) -> dict:
    """rank = posicion (1-based) del primer chunk esperado; rr = 1/rank (0 si no aparece)."""
    rank = next((i for i, cid in enumerate(retrieved, 1) if cid in expected), None)
    return {"rank": rank, "rr": 1 / rank if rank else 0.0, "hit@1": rank == 1, "hit@k": rank is not None}


def print_checks(checks: list[tuple[str, bool]], passed: bool, elapsed: float, error: str | None) -> None:
    if error:
        print(f"  ERROR: {error}")
    print("  Checks:")
    for name, ok in checks:
        print(f"    [{'OK' if ok else 'FAIL'}] {name}")
    print(f"  => {'PASS' if passed else 'FAIL'} {sum(ok for _, ok in checks)}/{len(checks)} | {elapsed:.1f}s")


def load_cases(path: Path, prefixes: list[str]) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if prefixes:
        cases = [c for c in cases if any(c["id"].startswith(p) for p in prefixes)]
    if not cases:
        raise SystemExit("Ningun caso coincide con el filtro.")
    return cases


def latency_by_node(results: list[dict]) -> dict[str, dict]:
    """Agrega latencia y tokens de las llamadas al LLM por nodo/agente a partir de los flujos."""
    groups: dict[str, list[dict]] = {}
    for r in results:
        for turn in r["turns"]:
            for step in turn["flow"]:
                if step["type"] in ("llm", "route"):
                    groups.setdefault(step.get("agent") or "agent", []).append(step)
    stats = {}
    for node, steps in groups.items():
        lat = sorted(s.get("latency_s") or 0.0 for s in steps)
        stats[node] = {
            "calls": len(steps),
            "latency_mean_s": round(sum(lat) / len(lat), 2),
            "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2),
            "latency_total_s": round(sum(lat), 1),
            "tokens_in_mean": round(sum(s.get("tokens_in") or 0 for s in steps) / len(steps)),
            "tokens_out_mean": round(sum(s.get("tokens_out") or 0 for s in steps) / len(steps)),
        }
    return stats


def summarize_and_save(prefix: str, meta: dict, results: list[dict], total_s: float) -> Path:
    print(f"\n{'=' * 72}\nRESUMEN | " + " ".join(f"{k}={v}" for k, v in meta.items()) + "\n")
    print(f"{'caso':<30}{'resultado':<11}{'checks':<9}tiempo")
    for r in results:
        score = f"{sum(c['ok'] for c in r['checks'])}/{len(r['checks'])}"
        status = "ERROR" if r["error"] else ("PASS" if r["passed"] else "FAIL")
        print(f"{r['id']:<30}{status:<11}{score:<9}{r['elapsed_s']}s")

    n_pass = sum(r["passed"] for r in results)
    n_checks = sum(len(r["checks"]) for r in results)
    ok_checks = sum(c["ok"] for r in results for c in r["checks"])
    print(f"\nCasos PASS: {n_pass}/{len(results)} | checks OK: {ok_checks}/{n_checks} | {total_s:.0f}s")

    nodes = latency_by_node(results)
    print(f"\n{'nodo':<16}{'llamadas':<10}{'lat media':<11}{'lat p95':<9}{'lat total':<11}tokens in/out (media)")
    for node, st in sorted(nodes.items(), key=lambda kv: -kv[1]["latency_total_s"]):
        print(f"{node:<16}{st['calls']:<10}{str(st['latency_mean_s']) + 's':<11}{str(st['latency_p95_s']) + 's':<9}"
              f"{str(st['latency_total_s']) + 's':<11}{st['tokens_in_mean']} / {st['tokens_out_mean']}")

    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.json"
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **meta,
        "app_today": get_settings().app_today.isoformat(),
        "summary": {"cases_passed": n_pass, "cases_total": len(results), "checks_ok": ok_checks,
                    "checks_total": n_checks, "elapsed_s": round(total_s, 1)},
        "latency_by_node": nodes,
        "cases": results,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Reporte: reports/{path.name}")
    return path
