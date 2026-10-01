"""Fase 1: evaluador de casos contra el agente real (Ollama).

Por cada caso muestra el flujo paso a paso (decision del LLM, tool calls, resultados,
razonamiento si esta activo, latencia y tokens), valida checks deterministas y guarda
el detalle completo en reports/. Es un evaluador heuristico; en la Fase 4 se reemplaza
por datasets en LangSmith + LLM-as-judge.

Uso:
    python scripts/fase1_eval.py                 # todos los casos
    python scripts/fase1_eval.py 01 13           # casos cuyo id empieza con 01 o 13
    python scripts/fase1_eval.py --reasoning     # activa thinking y muestra el razonamiento
    python scripts/fase1_eval.py --fallback      # usa OLLAMA_FALLBACK_MODEL
    python scripts/fase1_eval.py --quiet         # solo resumen
"""

import argparse
import time
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from agentic.agent import MAX_STEPS_MESSAGE, ToolCallingAgent
from agentic.domain.store import reset_store
from agentic.evalkit import llm_step, load_cases, print_checks, print_step, spec_checks, summarize_and_save, warm_up
from agentic.llm import get_chat_model
from agentic.tools import SHOP_TOOLS

CASES_FILE = Path(__file__).resolve().parents[1] / "evals" / "fase1_cases.json"


def extract_flow(messages: list) -> list[dict]:
    flow = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            flow.append(llm_step(msg))
        elif isinstance(msg, ToolMessage):
            flow.append({"type": "tool", "name": msg.name, "output": msg.content})
    return flow


def run_case(agent: ToolCallingAgent, case: dict, verbose: bool) -> dict:
    store = reset_store()
    history: list = []
    turns, checks = [], []
    error = None
    start = time.perf_counter()
    if verbose:
        print(f"\n=== [{case['id']}] {case['description']}")

    try:
        for i, turn in enumerate(case["turns"], 1):
            t0 = time.perf_counter()
            result = agent.run(
                turn["user"],
                history,
                langsmith_extra={"tags": ["fase1-eval"], "metadata": {"case_id": case["id"], "turn": i}},
            )
            elapsed = time.perf_counter() - t0
            flow = extract_flow(result.messages[len(history) + 2:])  # +2: system + mensaje del usuario
            history = result.messages[1:]
            called = [s["tool"] for s in result.steps]
            turn_checks = spec_checks(turn, result.answer, called, result.answer == MAX_STEPS_MESSAGE)
            checks += [(f"T{i} {name}", ok) for name, ok in turn_checks]
            turns.append({"user": turn["user"], "answer": result.answer, "elapsed_s": round(elapsed, 2), "flow": flow})
            if verbose:
                print(f"--- Turno {i} | Usuario: {turn['user']}")
                for step in flow:
                    print_step(step)
                print(f"  ShopAssist: {result.answer}\n  ({elapsed:.1f}s en el turno)")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluador de casos Fase 1")
    parser.add_argument("cases", nargs="*", help="prefijos de id de caso (ej. 01 13)")
    parser.add_argument("--reasoning", action="store_true", help="activa el modo thinking del modelo")
    parser.add_argument("--fallback", action="store_true", help="usa OLLAMA_FALLBACK_MODEL")
    parser.add_argument("--quiet", action="store_true", help="solo resumen")
    args = parser.parse_args()

    cases = load_cases(CASES_FILE, args.cases)
    overrides = {"reasoning": True} if args.reasoning else {}
    llm = get_chat_model(fallback=args.fallback, **overrides)
    agent = ToolCallingAgent(llm, SHOP_TOOLS)
    print(f"Modelo: {llm.model} | reasoning={llm.reasoning} | casos: {len(cases)}")
    warm_up(llm)

    start = time.perf_counter()
    results = [run_case(agent, case, verbose=not args.quiet) for case in cases]
    summarize_and_save("fase1_eval", {"model": llm.model, "reasoning": llm.reasoning}, results, time.perf_counter() - start)


if __name__ == "__main__":
    main()
