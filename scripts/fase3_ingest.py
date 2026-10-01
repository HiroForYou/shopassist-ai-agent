"""Fase 3: indexa knowledge/*.md en Qdrant (un chunk por seccion ##). Idempotente: recrea la coleccion.

Uso:
    python scripts/fase3_ingest.py
    python scripts/fase3_ingest.py --query "cuanto tarda el reembolso con tarjeta?"
"""

import argparse
import time

from agentic.config import get_settings
from agentic.rag import KnowledgeBase, load_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta de politicas en Qdrant")
    parser.add_argument("--query", default="Cuantos dias tengo para devolver un producto?", help="consulta de prueba")
    args = parser.parse_args()

    s = get_settings()
    chunks = load_chunks()
    print(f"Chunks: {len(chunks)} | embeddings: {s.ollama_embed_model} | Qdrant: {s.qdrant_url}/{s.qdrant_collection}\n")
    for c in chunks:
        print(f"  {c.chunk_id:<55} {len(c.text):>4} chars")

    kb = KnowledgeBase()
    start = time.perf_counter()
    kb.ingest(chunks)
    print(f"\nIndexados {len(chunks)} chunks en {time.perf_counter() - start:.1f}s")

    print(f"\nConsulta de prueba: {args.query!r}")
    for d in kb.search(args.query, score_threshold=0.0):
        print(f"  {d['metadata']['score']:.3f}  {d['metadata']['chunk_id']}")


if __name__ == "__main__":
    main()
