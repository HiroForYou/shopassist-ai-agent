"""Fase 3: RAG sobre las politicas de la tienda.

knowledge/*.md --(chunk por seccion ##)--> Chunk --(embeddings Ollama)--> Qdrant --(busqueda coseno)--> tool
"""

import re
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langsmith import traceable
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from agentic.config import get_settings
from agentic.observability import log_event

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<archivo>#<seccion>": se usa como cita
    source: str
    title: str
    section: str
    text: str

    def embedding_text(self) -> str:
        # El encabezado da contexto al fragmento (una seccion suelta pierde de que documento viene)
        return f"{self.title} - {self.section}\n{self.text}"


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def chunk_markdown(path: Path) -> list[Chunk]:
    """Un chunk por seccion `##`. El `#` del documento se usa como titulo."""
    title, section, lines, chunks = path.stem, None, [], []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if section and body:
            chunks.append(Chunk(f"{path.stem}#{slugify(section)}", path.name, title, section, body))

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            flush()
            section, lines = line[3:].strip(), []
        elif line.startswith("# "):
            title = line[2:].strip()
        elif section:
            lines.append(line)
    flush()
    return chunks


def load_chunks(directory: Path = KNOWLEDGE_DIR) -> list[Chunk]:
    return [c for path in sorted(directory.glob("*.md")) for c in chunk_markdown(path)]


class KnowledgeBase:
    def __init__(self, client: QdrantClient | None = None, embeddings=None, collection: str | None = None,
                 query_instruction: str | None = None):
        s = get_settings()
        self.client = client or QdrantClient(url=s.qdrant_url)
        self.embeddings = embeddings or OllamaEmbeddings(model=s.ollama_embed_model, base_url=s.ollama_base_url)
        self.collection = collection or s.qdrant_collection
        self.query_instruction = s.rag_query_instruction if query_instruction is None else query_instruction

    def ingest(self, chunks: list[Chunk]) -> int:
        """Recrea la coleccion: ingesta idempotente (mismos chunks -> mismos ids)."""
        vectors = self.embeddings.embed_documents([c.embedding_text() for c in chunks])
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
        )
        self.client.upsert(self.collection, points=[
            PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)), vector=v, payload=asdict(c))
            for c, v in zip(chunks, vectors)
        ])
        return len(chunks)

    @traceable(name="policy_retriever", run_type="retriever")
    def search(self, query: str, k: int | None = None, score_threshold: float | None = None) -> list[dict]:
        """Devuelve documentos en formato retriever de LangSmith (page_content + metadata)."""
        s = get_settings()
        start = time.perf_counter()
        vector = self.embeddings.embed_query(f"{self.query_instruction}{query}")
        points = self.client.query_points(
            self.collection,
            query=vector,
            limit=k or s.rag_top_k,
            score_threshold=s.rag_score_threshold if score_threshold is None else score_threshold,
            with_payload=True,
        ).points
        log_event("retrieval", latency_s=round(time.perf_counter() - start, 3), k=k or s.rag_top_k,
                  results=len(points), top_score=round(points[0].score, 3) if points else None)
        return [
            {
                "page_content": p.payload["text"],
                "type": "Document",
                "metadata": {**{key: v for key, v in p.payload.items() if key != "text"}, "score": round(p.score, 3)},
            }
            for p in points
        ]


def make_search_tool(kb: KnowledgeBase):
    @tool
    def search_policies(query: str) -> dict:
        """Busca en las politicas de la tienda: reembolsos, envios, garantia, pagos y atencion al cliente.
        `query`: la pregunta del cliente en lenguaje natural."""
        docs = kb.search(query)
        if not docs:
            return {"results": [], "note": "Sin resultados relevantes en las politicas."}
        return {"results": [
            {"source": d["metadata"]["chunk_id"], "score": d["metadata"]["score"], "text": d["page_content"]}
            for d in docs
        ]}

    return search_policies
