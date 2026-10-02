from datetime import date
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# LangSmith lee LANGSMITH_* directamente de os.environ, por eso se carga el .env al proceso.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"
    ollama_fallback_model: str = "granite4.1:3b"
    ollama_temperature: float = 0.0
    ollama_reasoning: bool = False
    ollama_embed_model: str = "qwen3-embedding:0.6b"

    # LLM-as-judge (Fase 4). Vacio = mismo modelo que el agente (ojo: sesgo de auto-preferencia)
    judge_model: str = ""
    judge_reasoning: bool = False
    judge_max_tokens: int = 600  # num_predict del juez: 3 frases + issues + JSON caben con margen

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "shop_policies"
    rag_top_k: int = 3
    # Similitud coseno minima (calibrada con scripts/fase3_eval.py retrieval): descarta preguntas ajenas a la tienda
    # sin perder chunks esperados. Las preguntas sin respuesta del dominio no son separables: las maneja el prompt.
    rag_score_threshold: float = 0.40
    # qwen3-embedding mejora con una instruccion en la consulta (no en los documentos)
    rag_query_instruction: str = (
        "Instruct: Given a customer question, retrieve the store policy passages that answer it\nQuery: "
    )

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "ai-agentic"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # Observabilidad (Fase 5)
    obs_enabled: bool = True
    # Tarifa HIPOTETICA para estimar costo como si fuera una API de pago (USD por 1M tokens).
    # Con Ollama el costo real es infraestructura (CPU/RAM/hora), no tokens. Ajustar al proveedor real.
    cost_input_per_1m: float = 1.0
    cost_output_per_1m: float = 4.0
    # Router: "hybrid" (reglas deterministas + LLM como fallback; adoptado en Fase 5: -20% latencia, -19% tokens,
    # 0 regresiones en A/B misma sesion) | "llm" (solo LLM, baseline Fase 4)
    router_mode: str = "hybrid"

    # Guardrails y resiliencia (Fase 6)
    guardrails_enabled: bool = True
    llm_timeout_s: float = 240.0  # CPU: p95 por llamada ~60 s en Fase 5; el timeout corta cuelgues, no llamadas lentas
    llm_retries: int = 1  # reintentos sobre el modelo principal antes del fallback

    # API (Fase 7)
    api_warmup: bool = True  # carga los modelos al arrancar: el primer usuario no paga el arranque en frio
    api_max_concurrency: int = 1  # Ollama en CPU: turnos serializados; el resto espera en cola (metrica)
    api_allow_reset: bool = True  # /admin/reset para benchmarks locales; false en cualquier entorno compartido

    app_today: date = date(2026, 9, 24)


@lru_cache
def get_settings() -> Settings:
    return Settings()
