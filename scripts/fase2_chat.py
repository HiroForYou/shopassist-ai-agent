"""Fase 2-3: chat por consola con el sistema multi-agent (incluye policy_agent con RAG). Flujo nodo a nodo en vivo.
Requiere haber indexado las politicas (scripts/fase3_ingest.py) para las preguntas de politicas.

Uso:
    python scripts/fase2_chat.py
Comandos: /nuevo (nueva conversacion), /estado (estado del grafo), salir
"""

from agentic.evalkit import print_step
from agentic.llm import get_resilient_chat_model
from agentic.multiagent import ShopAssistChat, build_graph


def main() -> None:
    chat = ShopAssistChat(build_graph(get_resilient_chat_model()))
    thread = chat.new_thread()
    print(f"ShopAssist multi-agent | thread {thread}\nComandos: /nuevo, /estado, salir\n")

    while (user := input("Tu: ").strip()).lower() not in {"salir", "exit"}:
        if not user:
            continue
        if user == "/nuevo":
            thread = chat.new_thread()
            print(f"Nueva conversacion: thread {thread}\n")
            continue
        if user == "/estado":
            values = chat.graph.get_state({"configurable": {"thread_id": thread}}).values
            print(f"  thread={thread} | agente activo={values.get('active_agent')} | "
                  f"mensajes={len(values.get('messages', []))}\n")
            continue

        result = chat.send(thread, user, on_step=print_step, tags=["fase2-chat"])
        print(f"ShopAssist: {result.answer}\n  ({result.elapsed_s}s)\n")


if __name__ == "__main__":
    main()
