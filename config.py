"""Wspolne stale konfiguracyjne dla calego systemu."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
QDRANT_PATH = str(DATA_DIR / "qdrant_db")
BM25_PATH = DATA_DIR / "bm25_index.pkl"

# Qdrant
COLLECTION = "legal_pl"

# Embeddings
EMBED_MODEL = "intfloat/multilingual-e5-base"
EMBED_DIM = 768
EMBED_BATCH = 64

# Retrieval
TOP_K = 5            # ile chunkow trafia do kontekstu Gemini
RRF_K = 60           # stala RRF
CANDIDATES = 30      # ile kandydatow z kazdej listy (BM25 / semantic) przed fuzja

# Rejestr kodeksow: SKROT -> (adres ELI w ISAP, pelna nazwa, ranga hierarchii)
# Adresy to akty bazowe; ISAP serwuje ich skonsolidowany tekst ujednolicony pod /text.html.
KODEKSY = {
    "KC":  {"address": "DU/1964/93",   "full": "Kodeks cywilny",                       "rank": 2},
    "KPC": {"address": "DU/1964/296",  "full": "Kodeks postepowania cywilnego",        "rank": 2},
    "KK":  {"address": "DU/1997/553",  "full": "Kodeks karny",                         "rank": 2},
    "KPK": {"address": "DU/1997/555",  "full": "Kodeks postepowania karnego",          "rank": 2},
    "KP":  {"address": "DU/1974/141",  "full": "Kodeks pracy",                         "rank": 2},
    "KPA": {"address": "DU/1960/168",  "full": "Kodeks postepowania administracyjnego","rank": 2},
    "KSH": {"address": "DU/2000/1037", "full": "Kodeks spolek handlowych",             "rank": 2},
}

# Konserwatywne minima do sanity-checku chunkera (wykrywaja pobranie fragmentu
# zamiast pelnego tekstu jednolitego). Wartosci dobrane do RZECZYWISTEJ liczby
# artykulow w mocy w tekscie skonsolidowanym ISAP (uchylone artykuly sa pomijane).
MINIMUM_CHUNKS = {
    "KC":  900,
    "KPC": 1000,
    "KK":  300,
    "KPK": 500,
    "KP":  200,
    "KPA": 150,   # tekst skonsolidowany ma ~196 art. w mocy (uchylone usuniete)
    "KSH": 500,
}
