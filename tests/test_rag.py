"""Tests de RAG sin servidores: Qdrant en memoria + embeddings deterministas por bolsa de palabras."""

import hashlib
import math

import pytest
from qdrant_client import QdrantClient

from agentic.evalkit import normalize, retrieval_metrics
from agentic.rag import KnowledgeBase, chunk_markdown, load_chunks, make_search_tool, slugify

DIM = 256


class BagOfWordsEmbeddings:
    """Vector = conteo de palabras hasheadas. Suficiente para verificar ingesta y busqueda."""

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in normalize(text).split():
            word = word.strip("?.,;:!()")
            if len(word) > 3:
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def kb() -> KnowledgeBase:
    kb = KnowledgeBase(client=QdrantClient(":memory:"), embeddings=BagOfWordsEmbeddings(),
                       collection="test", query_instruction="")
    kb.ingest(load_chunks())
    return kb


def test_slugify():
    assert slugify("Tiempo de acreditación del reembolso") == "tiempo-de-acreditacion-del-reembolso"


def test_chunk_markdown_one_chunk_per_section(tmp_path):
    doc = tmp_path / "demo.md"
    doc.write_text("# Titulo\n\nintro ignorada\n\n## Uno\nTexto uno.\n\n## Dos\nTexto dos.\n## Vacia\n", encoding="utf-8")
    chunks = chunk_markdown(doc)

    assert [c.chunk_id for c in chunks] == ["demo#uno", "demo#dos"]
    assert chunks[0].title == "Titulo"
    assert chunks[0].embedding_text().startswith("Titulo - Uno\n")


def test_knowledge_base_ids_are_unique():
    ids = [c.chunk_id for c in load_chunks()]
    assert len(ids) == len(set(ids)) > 0


def test_search_returns_expected_chunk(kb):
    docs = kb.search("costo de envio en provincias", k=3, score_threshold=0.0)

    assert docs[0]["metadata"]["chunk_id"] == "envios#costo-de-envio"
    assert docs[0]["page_content"]
    assert 0 < docs[0]["metadata"]["score"] <= 1


def test_ingest_is_idempotent(kb):
    kb.ingest(load_chunks())
    assert kb.client.count("test").count == len(load_chunks())


def test_score_threshold_filters(kb):
    assert kb.search("receta ceviche limon", score_threshold=0.5) == []


def test_search_tool_formats_sources(kb):
    tool = make_search_tool(kb)
    out = tool.invoke({"query": "garantia electronica meses"})

    assert out["results"][0]["source"].startswith("garantia#")
    assert {"source", "score", "text"} <= set(out["results"][0])


def test_search_tool_reports_empty(kb, monkeypatch):
    monkeypatch.setattr(kb, "search", lambda query: [])
    out = make_search_tool(kb).invoke({"query": "nada"})

    assert out["results"] == []
    assert "note" in out


@pytest.mark.parametrize(
    "retrieved, rank, rr",
    [(["a", "b"], 1, 1.0), (["x", "a"], 2, 0.5), (["x", "y"], None, 0.0)],
)
def test_retrieval_metrics(retrieved, rank, rr):
    m = retrieval_metrics(["a"], retrieved)
    assert (m["rank"], m["rr"]) == (rank, rr)
