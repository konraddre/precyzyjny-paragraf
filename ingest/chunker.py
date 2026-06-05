"""Chunking per jednostka redakcyjna (artykuł).

Tekst wejściowy (data/raw/{SKROT}.txt) ma już jeden artykuł na akapit, każdy
zaczyna się od wiersza "Art. N." (N może mieć indeks górny, np. "22^1", i literę,
np. "471a"). Dzielimy PER ARTYKUŁ — nigdy po liczbie znaków/tokenów. Artykuły
dłuższe niż ~800 tokenów dzielimy dalej na poziomie paragrafów (§), zachowując
nagłówek artykułu w każdym sub-chunku.
"""
import re

# Granica artykułu: początek wiersza "Art. N." (obsługuje 471a oraz 22^1).
ARTICLE_PATTERN = re.compile(r"(?=^\s*Art\.\s+\d+(?:\^\d+)?[a-z]?\.)", re.MULTILINE)
ARTICLE_NO = re.compile(r"^\s*Art\.\s+(\d+(?:\^\d+)?[a-z]?)\.")
PARA_SPLIT = re.compile(r"(?=§\s*\d+(?:\^\d+)?[a-z]?\.)")

LONG_ARTICLE_WORDS = 600  # ~800 tokenów


def _make_chunk(text, article, source, source_full, rank, last_updated):
    return {
        "text": text.strip(),
        "source": source,
        "source_full": source_full,
        "article": article,
        "hierarchy_rank": rank,
        "last_updated": last_updated,
    }


def chunk_by_article(text: str, source: str, source_full: str,
                     hierarchy_rank: int, last_updated: str) -> list[dict]:
    chunks: list[dict] = []
    blocks = [b.strip() for b in ARTICLE_PATTERN.split(text) if b.strip()]

    for block in blocks:
        m = ARTICLE_NO.match(block)
        if not m:
            continue  # tekst przed pierwszym artykułem (tytuł aktu) — pomijamy
        article = m.group(1)
        header = f"Art. {article}."

        if len(block.split()) <= LONG_ARTICLE_WORDS:
            chunks.append(_make_chunk(block, article, source, source_full,
                                      hierarchy_rank, last_updated))
            continue

        # Długi artykuł -> podział na poziomie paragrafów, nagłówek w każdym sub-chunku.
        body = block[m.end():].strip()
        parts = [p.strip() for p in PARA_SPLIT.split(body) if p.strip()]
        if len(parts) <= 1:
            chunks.append(_make_chunk(block, article, source, source_full,
                                      hierarchy_rank, last_updated))
            continue
        for part in parts:
            sub = part if part.startswith(header) else f"{header} {part}"
            chunks.append(_make_chunk(sub, article, source, source_full,
                                      hierarchy_rank, last_updated))

    return chunks
