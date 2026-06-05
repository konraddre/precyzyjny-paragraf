"""Wrapper na sentence-transformers (intfloat/multilingual-e5-base).

Model e5 wymaga prefiksów: 'passage: ' przy indeksowaniu, 'query: ' przy zapytaniu.
Embeddingi normalizowane (cosine). Wymiar 768.
"""
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import EMBED_MODEL, EMBED_BATCH  # noqa: E402


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        vecs = self.model.encode(
            prefixed,
            batch_size=EMBED_BATCH,
            normalize_embeddings=True,
            show_progress_bar=len(prefixed) > 256,
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, query: str) -> list[float]:
        vec = self.model.encode(
            f"query: {query}",
            normalize_embeddings=True,
        )
        return vec.tolist()
