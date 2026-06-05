"""Hybrydowy retrieval: BM25 + semantic, łączone przez Reciprocal Rank Fusion.

Zwraca top-5 chunków do kontekstu Gemini. Kontekst ma być precyzyjny, nie obszerny.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TOP_K, RRF_K, CANDIDATES, BM25_PATH  # noqa: E402
from retrieval.bm25_index import BM25Index  # noqa: E402
from retrieval.embedder import Embedder  # noqa: E402
from retrieval import vector_store  # noqa: E402


def reciprocal_rank_fusion(bm25_hits: list[dict], semantic_hits: list[dict],
                           k: int = RRF_K) -> list[dict]:
    """Standardowy RRF. Każdy chunk: score = sum(1 / (k + rank)) ze wszystkich list.

    Fuzja po stabilnym 'id' chunku. Zwraca listę chunków posortowaną malejąco.
    """
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, dict] = {}
    for hits in (bm25_hits, semantic_hits):
        for rank, chunk in enumerate(hits, start=1):
            cid = chunk.get("id") or chunk["text"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_by_id.setdefault(cid, chunk)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [chunk_by_id[cid] for cid in ordered]


class HybridSearcher:
    def __init__(self, embedder: Embedder, bm25: BM25Index):
        self.embedder = embedder
        self.bm25 = bm25

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        bm25_hits = self.bm25.search(query, CANDIDATES)
        qvec = self.embedder.embed_query(query)
        semantic_hits = vector_store.search_semantic(qvec, CANDIDATES)
        fused = reciprocal_rank_fusion(bm25_hits, semantic_hits)
        return fused[:top_k]


def load_searcher() -> HybridSearcher:
    """Buduje searcher z zapisanego indeksu BM25 + modelu embeddingowego."""
    bm25 = BM25Index.load(BM25_PATH) if Path(BM25_PATH).exists() else BM25Index([])
    return HybridSearcher(Embedder(), bm25)
