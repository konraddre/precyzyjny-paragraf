"""Qdrant w trybie lokalnym (bez serwera). Inicjalizacja idempotentna.

NIGDY nie używamy recreate_collection — skasowałoby całą bazę przy każdym starcie.
Zamiast tego create_collection tylko gdy kolekcja nie istnieje + pomocniki delta
per kodeks (is_indexed / delete_kodeks).
"""
import atexit
import sys
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, Filter, FieldCondition, MatchValue,
    FilterSelector, PointStruct,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import COLLECTION, QDRANT_PATH, EMBED_DIM  # noqa: E402

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
        atexit.register(_close_client)  # czyste zamknięcie lokalnego storage
    return _client


def _close_client() -> None:
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


def ensure_collection() -> None:
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def _source_filter(source_code: str) -> Filter:
    return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_code))])


def is_indexed(source_code: str) -> bool:
    client = get_client()
    results, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=_source_filter(source_code),
        limit=1,
    )
    return len(results) > 0


def delete_kodeks(source_code: str) -> None:
    client = get_client()
    client.delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(filter=_source_filter(source_code)),
    )


def embed_and_store(chunks: list[dict], embedder) -> None:
    """Liczy embeddingi (batchami w embedderze) i zapisuje punkty do Qdrant.

    Każdemu chunkowi nadaje stabilne 'id' (UUID) — używane jako id punktu oraz do
    fuzji RRF. Pełne metadane trafiają do payloadu.
    """
    if not chunks:
        return
    client = get_client()
    vectors = embedder.embed_passages([c["text"] for c in chunks])
    points = []
    for chunk, vec in zip(chunks, vectors):
        chunk.setdefault("id", str(uuid.uuid4()))
        points.append(PointStruct(id=chunk["id"], vector=vec, payload=chunk))
    # Upsert partiami, by nie trzymać wszystkiego naraz.
    for i in range(0, len(points), 256):
        client.upsert(collection_name=COLLECTION, points=points[i:i + 256])


def load_chunks_from_qdrant(source_code: str) -> list[dict]:
    """Odtwarza listę chunków (payload bez wektorów) dla danego kodeksu — do BM25."""
    client = get_client()
    out, offset = [], None
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=_source_filter(source_code),
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        out.extend(r.payload for r in results)
        if offset is None:
            break
    return out


def search_semantic(query_vector: list[float], limit: int) -> list[dict]:
    client = get_client()
    hits = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    ).points
    return [h.payload for h in hits]


def count_by_source() -> dict[str, int]:
    """Liczba chunków per kodeks — do paska bocznego UI."""
    from config import KODEKSY
    client = get_client()
    counts = {}
    for skrot in KODEKSY:
        try:
            counts[skrot] = client.count(
                collection_name=COLLECTION,
                count_filter=_source_filter(skrot),
            ).count
        except Exception:
            counts[skrot] = 0
    return counts
