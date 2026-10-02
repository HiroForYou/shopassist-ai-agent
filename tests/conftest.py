import os

# Los tests unitarios no envian trazas a LangSmith (load_dotenv no sobrescribe variables ya definidas).
os.environ["LANGSMITH_TRACING"] = "false"
# Ni logs a logs/shopassist.jsonl: los tests que verifican eventos usan un sink en memoria.
os.environ["OBS_ENABLED"] = "false"
# Los LLM guionados incluyen la decision del router en la cola: modo llm por defecto en tests.
# Los tests del router hibrido pasan router_mode="hybrid" explicitamente.
os.environ["ROUTER_MODE"] = "llm"

import pytest  # noqa: E402

from agentic.domain.store import reset_store  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_store():
    return reset_store()
