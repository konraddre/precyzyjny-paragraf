"""BM25 z prostą polską tokenizacją.

Indeks trzyma równolegle: tokenizowany korpus (dla BM25Okapi) oraz listę chunków
(z polami metadanych i 'id'), w tej samej kolejności. search() zwraca chunki.
"""
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

POLISH_STOPWORDS = {
    "i", "w", "z", "do", "na", "się", "że", "to", "jest", "są",
    "przez", "po", "o", "a", "ze", "jak", "ten", "ta", "przy", "dla",
    "lub", "oraz", "też", "już", "być", "co", "go", "jej", "jego",
    # UWAGA: "nie" celowo pominięte — negacja zmienia sens przepisu prawnego
    # ("Art. 471 KC nie stosuje się" != "Art. 471 KC stosuje się").
}

_TOKEN_STRIP = ".,;:()[]§"


def tokenize_pl(text: str) -> list[str]:
    tokens = text.lower().split()
    return [t.strip(_TOKEN_STRIP) for t in tokens
            if t not in POLISH_STOPWORDS and len(t.strip(_TOKEN_STRIP)) > 2]


class BM25Index:
    def __init__(self, chunks: list[dict] | None = None):
        self.chunks: list[dict] = chunks or []
        self.corpus = [tokenize_pl(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def search(self, query: str, top_n: int) -> list[dict]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize_pl(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:top_n] if scores[i] > 0]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks}, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["chunks"])
