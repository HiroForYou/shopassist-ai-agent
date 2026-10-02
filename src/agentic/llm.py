import json
import urllib.request

from langchain_ollama import ChatOllama

from agentic.config import get_settings


def get_chat_model(fallback: bool = False, **overrides) -> ChatOllama:
    s = get_settings()
    params = {
        "model": s.ollama_fallback_model if fallback else s.ollama_model,
        "base_url": s.ollama_base_url,
        "temperature": s.ollama_temperature,
        "reasoning": s.ollama_reasoning,
        "client_kwargs": {"timeout": s.llm_timeout_s},  # sin timeout, un Ollama colgado bloquea el turno para siempre
    }
    params.update(overrides)
    return ChatOllama(**params)


def get_resilient_chat_model(**overrides):
    """Modelo principal con reintento y fallback a OLLAMA_FALLBACK_MODEL (Fase 6)."""
    from agentic.guardrails import ResilientChatModel

    s = get_settings()
    return ResilientChatModel(get_chat_model(**overrides), get_chat_model(fallback=True, **overrides),
                              retries=s.llm_retries)


def model_digests(names: list[str]) -> dict[str, str]:
    """Digest corto (12 hex) de cada modelo segun Ollama. Un tag como `qwen3.5:4b` puede cambiar de pesos al hacer
    `ollama pull`; el digest identifica la version exacta y se registra en cada experimento."""
    try:
        with urllib.request.urlopen(f"{get_settings().ollama_base_url}/api/tags", timeout=5) as resp:
            models = {m["name"]: m.get("digest", "") for m in json.load(resp)["models"]}
    except Exception:
        return {name: "desconocido" for name in names}
    return {name: (models.get(name) or "no-descargado")[:12] for name in names}
