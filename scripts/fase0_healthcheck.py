"""Fase 0: verifica que Ollama, el modelo, el tool calling y LangSmith esten operativos."""

import json
import os
import time
import urllib.request

from langchain_core.tools import tool

from agentic.config import get_settings
from agentic.llm import get_chat_model


def report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -> {detail}" if detail else ""))
    return ok


def check_ollama(s) -> bool:
    try:
        with urllib.request.urlopen(f"{s.ollama_base_url}/api/tags", timeout=5) as resp:
            models = [m["name"] for m in json.load(resp)["models"]]
    except Exception as exc:
        return report("Ollama accesible", False, f"{s.ollama_base_url}: {exc}")
    report("Ollama accesible", True, s.ollama_base_url)
    main_ok = report("Modelo principal", s.ollama_model in models, s.ollama_model)
    report("Modelo fallback", s.ollama_fallback_model in models, s.ollama_fallback_model)  # opcional hasta Fase 6
    report("Modelo embeddings", s.ollama_embed_model in models, s.ollama_embed_model)  # Fase 3
    return main_ok


def check_qdrant(s) -> bool:
    try:
        with urllib.request.urlopen(f"{s.qdrant_url}/collections", timeout=5) as resp:
            names = [c["name"] for c in json.load(resp)["result"]["collections"]]
    except Exception as exc:
        return report("Qdrant", False, f"{s.qdrant_url}: {exc}")
    indexed = s.qdrant_collection in names
    detail = f"coleccion '{s.qdrant_collection}'" + ("" if indexed else " sin indexar: python scripts/fase3_ingest.py")
    return report("Qdrant", indexed, detail)


def check_chat() -> bool:
    start = time.perf_counter()
    reply = get_chat_model().invoke("Responde solo con la palabra: OK")
    return report("Chat", bool(reply.content), f"{reply.content!r} en {time.perf_counter() - start:.1f}s")


@tool
def get_weather(city: str) -> str:
    """Devuelve el clima de una ciudad."""
    return f"Soleado en {city}"


def check_tool_calling() -> bool:
    reply = get_chat_model().bind_tools([get_weather]).invoke("Que clima hace en Lima?")
    return report("Tool calling", bool(reply.tool_calls), str(reply.tool_calls))


def check_langsmith(s) -> bool:
    if not s.langsmith_tracing:
        return report("LangSmith", False, "LANGSMITH_TRACING=false (trazas desactivadas)")
    if not s.langsmith_api_key or s.langsmith_api_key.startswith("lsv2_pt_xxx"):
        return report("LangSmith", False, "LANGSMITH_API_KEY no configurada en .env")
    try:
        from langsmith import Client

        next(iter(Client().list_projects(limit=1)), None)
    except Exception as exc:
        return report("LangSmith", False, str(exc))
    return report("LangSmith", True, f"proyecto '{os.getenv('LANGSMITH_PROJECT', s.langsmith_project)}'")


def main() -> None:
    s = get_settings()
    print(f"Modelo: {s.ollama_model} (reasoning={s.ollama_reasoning}) | Ollama: {s.ollama_base_url}\n")
    if not check_ollama(s):
        raise SystemExit(1)
    results = [check_chat(), check_tool_calling(), check_langsmith(s), check_qdrant(s)]
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
