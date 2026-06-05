"""Skrypt ingestion (uruchamiany jednorazowo i przy aktualizacji bazy).

    python ingest/build_index.py              # pomija kodeksy już zaindeksowane
    python ingest/build_index.py --force      # re-indeksuje wszystko od zera
    python ingest/build_index.py --force KC KPC   # re-indeksuje tylko wymienione

Logika delta: nie kasuje istniejącej bazy. Jeśli pobranie jednego kodeksu się nie
powiedzie, loguje błąd i kontynuuje z pozostałymi (poprzednie dane zachowane).
BM25 jest przebudowywany ZAWSZE z pełnej bazy (nie tylko nowo dodanych chunków).
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import KODEKSY, MINIMUM_CHUNKS, BM25_PATH  # noqa: E402
from ingest.isap_client import fetch_from_isap  # noqa: E402
from ingest.chunker import chunk_by_article  # noqa: E402
from retrieval.embedder import Embedder  # noqa: E402
from retrieval.bm25_index import BM25Index  # noqa: E402
from retrieval import vector_store as vs  # noqa: E402


def validate_chunks(skrot: str, chunks: list) -> None:
    minimum = MINIMUM_CHUNKS.get(skrot, 50)
    if len(chunks) < minimum:
        raise ValueError(
            f"[{skrot}] Walidacja nieudana: {len(chunks)} chunków < minimum {minimum}.\n"
            f"Prawdopodobna przyczyna: ISAP zwrócił fragment zamiast tekstu jednolitego,\n"
            f"lub parser nie rozpoznał formatu artykułów. Sprawdź data/raw/{skrot}.txt."
        )
    print(f"  ✓ {skrot}: {len(chunks)} chunków (minimum: {minimum}) — OK")


def main():
    parser = argparse.ArgumentParser(description="Ingestion polskich kodeksów do RAG.")
    parser.add_argument("--force", action="store_true",
                        help="Re-indeksuj nawet jeśli kodeks już w bazie.")
    parser.add_argument("kodeksy", nargs="*",
                        help="Skróty do re-indeksowania (puste = wszystkie).")
    args = parser.parse_args()

    unknown = [k for k in args.kodeksy if k not in KODEKSY]
    if unknown:
        parser.error(f"Nieznane skróty kodeksów: {unknown}. Dostępne: {list(KODEKSY)}")

    vs.ensure_collection()
    embedder = Embedder()
    all_chunks: list[dict] = []

    for skrot, meta in KODEKSY.items():
        force_this = args.force and (not args.kodeksy or skrot in args.kodeksy)

        if vs.is_indexed(skrot) and not force_this:
            print(f"  — {skrot}: już zaindeksowany, pomijam (użyj --force aby wymusić)")
            all_chunks.extend(vs.load_chunks_from_qdrant(skrot))
            continue

        print(f"  → {skrot}: pobieranie z ISAP...")
        try:
            text, last_updated = fetch_from_isap(skrot)
            chunks = chunk_by_article(text, skrot, meta["full"], meta["rank"], last_updated)
            validate_chunks(skrot, chunks)
        except Exception as e:
            print(f"  ✗ {skrot}: BŁĄD — {e}. Pomijam, poprzednie dane zachowane.")
            all_chunks.extend(vs.load_chunks_from_qdrant(skrot))
            continue

        if force_this and vs.is_indexed(skrot):
            vs.delete_kodeks(skrot)

        vs.embed_and_store(chunks, embedder)
        all_chunks.extend(chunks)
        print(f"  ✓ {skrot}: zaindeksowano {len(chunks)} artykułów (stan na {last_updated})")

    # BM25 zawsze z PEŁNEJ bazy.
    print("  → przebudowa indeksu BM25...")
    BM25Index(all_chunks).save(BM25_PATH)
    print(f"\nGotowe. Łącznie {len(all_chunks)} chunków w bazie.")


if __name__ == "__main__":
    main()
