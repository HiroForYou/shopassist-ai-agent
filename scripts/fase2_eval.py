"""Fase 2: evaluador de casos contra el sistema multi-agent (Ollama).

Ademas de los checks de Fase 1 valida el routing (`route_any`) y registra agentes y handoffs por turno.

Uso:
    python scripts/fase2_eval.py                 # todos los casos
    python scripts/fase2_eval.py 03 09           # casos cuyo id empieza con 03 o 09
    python scripts/fase2_eval.py --reasoning     # activa thinking y muestra el razonamiento
    python scripts/fase2_eval.py --fallback      # usa OLLAMA_FALLBACK_MODEL
    python scripts/fase2_eval.py --quiet         # solo resumen
"""

import argparse
import time
from pathlib import Path

from agentic.evalkit import load_cases, summarize_and_save, warm_up
from agentic.graph_eval import run_graph_case
from agentic.llm import get_chat_model
from agentic.multiagent import ShopAssistChat, build_graph
from agentic.rag import KnowledgeBase

CASES_FILE = Path(__file__).resolve().parents[1] / "evals" / "fase2_cases.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluador de casos Fase 2 (multi-agent)")
    parser.add_argument("cases", nargs="*", help="prefijos de id de caso (ej. 03 09)")
    parser.add_argument("--reasoning", action="store_true", help="activa el modo thinking del modelo")
    parser.add_argument("--fallback", action="store_true", help="usa OLLAMA_FALLBACK_MODEL")
    parser.add_argument("--quiet", action="store_true", help="solo resumen")
    args = parser.parse_args()

    cases = load_cases(CASES_FILE, args.cases)
    overrides = {"reasoning": True} if args.reasoning else {}
    llm = get_chat_model(fallback=args.fallback, **overrides)
    kb = KnowledgeBase()
    chat = ShopAssistChat(build_graph(llm, kb=kb))
    print(f"Modelo: {llm.model} | reasoning={llm.reasoning} | casos: {len(cases)}")
    warm_up(llm, kb.embeddings)

    start = time.perf_counter()
    results = [run_graph_case(chat, case, not args.quiet, "fase2-eval") for case in cases]
    summarize_and_save("fase2_eval", {"model": llm.model, "reasoning": llm.reasoning}, results, time.perf_counter() - start)


if __name__ == "__main__":
    main()
