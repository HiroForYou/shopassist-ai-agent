from langchain_ollama import ChatOllama

from agentic.config import get_settings


def get_chat_model(fallback: bool = False, **overrides) -> ChatOllama:
    s = get_settings()
    params = {
        "model": s.ollama_fallback_model if fallback else s.ollama_model,
        "base_url": s.ollama_base_url,
        "temperature": s.ollama_temperature,
        "reasoning": s.ollama_reasoning,
    }
    params.update(overrides)
    return ChatOllama(**params)
