"""Runner de casos multi-turno contra el grafo multi-agent (Fases 2+). Checks heuristicos por turno."""

import time

from agentic.domain.store import reset_store
from agentic.evalkit import print_checks, print_step, turn_checks
from agentic.multiagent import ROUTE_TO_AGENT, ShopAssistChat

AGENT_TO_ROUTE = {agent: route for route, agent in ROUTE_TO_AGENT.items()} | {"__end__": "general"}


def tool_outputs(flow: list[dict], name: str) -> list[str]:
    return [s["output"] for s in flow if s["type"] == "tool" and s["name"] == name]


def run_graph_case(chat: ShopAssistChat, case: dict, verbose: bool, tag: str) -> dict:
    """Ejecuta los turnos de un caso en un thread nuevo y valida:
    route_any, must_call, must_not_call, mention_any/all, cites_any (fuentes RAG citadas) y reembolsos creados."""
    store = reset_store()
    thread = f"{tag}-{case['id']}-{chat.new_thread()}"
    turns, checks = [], []
    error = None
    start = time.perf_counter()
    if verbose:
        print(f"\n=== [{case['id']}] {case['description']}")

    try:
        for i, turn in enumerate(case["turns"], 1):
            if verbose:
                print(f"--- Turno {i} | Usuario: {turn['user']}")
            result = chat.send(
                thread, turn["user"],
                on_step=print_step if verbose else None,
                tags=[tag], metadata={"case_id": case["id"], "turn": i},
            )
            route = AGENT_TO_ROUTE.get(result.route, result.route)
            retrieved = " ".join(tool_outputs(result.flow, "search_policies"))
            checks += [(f"T{i} {name}", ok) for _, name, ok in
                       turn_checks(turn, result.answer, result.tools_called, route, retrieved, result.hit_limit)]
            handoffs = [t for t in result.tools_called if t.startswith("transfer_to_")]
            turns.append({"user": turn["user"], "answer": result.answer, "route": route, "agents": result.agents,
                          "handoffs": handoffs, "elapsed_s": result.elapsed_s, "flow": result.flow})
            if verbose:
                extra = f" | handoffs: {handoffs}" if handoffs else ""
                print(f"  ShopAssist: {result.answer}\n  ({result.elapsed_s}s | agentes: {result.agents}{extra})")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        checks.append(("ejecucion sin errores", False))

    statuses = [r.status for r in store.refunds.values()]
    checks.append((f"reembolsos creados {statuses} en {case['refunds']}", statuses in case["refunds"]))
    passed = all(ok for _, ok in checks)
    total = time.perf_counter() - start
    if verbose:
        print_checks(checks, passed, total, error)

    return {
        "id": case["id"],
        "description": case["description"],
        "passed": passed,
        "error": error,
        "elapsed_s": round(total, 2),
        "refunds": statuses,
        "checks": [{"name": n, "ok": ok} for n, ok in checks],
        "turns": turns,
    }
