"""Fase 1: chat por consola con el agente ShopAssist.

Uso:
    python scripts/fase1_agent.py                          # modo interactivo
    python scripts/fase1_agent.py "Quiero devolver el pedido A1001"
"""

import sys

from agentic.agent import ToolCallingAgent
from agentic.llm import get_chat_model
from agentic.tools import SHOP_TOOLS


def print_result(result) -> None:
    for step in result.steps:
        print(f"  -> {step['tool']}({step['args']}) = {step['output']}")
    print(f"ShopAssist: {result.answer}\n")


def main() -> None:
    agent = ToolCallingAgent(get_chat_model(), SHOP_TOOLS)

    if len(sys.argv) > 1:
        print_result(agent.run(" ".join(sys.argv[1:])))
        return

    history = []
    print("ShopAssist (escribe 'salir' para terminar)\n")
    while (user := input("Tu: ").strip()).lower() not in {"salir", "exit"}:
        if not user:
            continue
        result = agent.run(user, history)
        history = result.messages[1:]  # se descarta el system prompt; el agente lo agrega en cada run
        print_result(result)


if __name__ == "__main__":
    main()
