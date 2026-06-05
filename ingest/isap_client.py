"""Klient ISAP / ELI API (api.sejm.gov.pl).

Pobiera SKONSOLIDOWANY tekst jednolity kodeksu (PDF typu "U" = ujednolicony)
i konwertuje go na czysty tekst, w którym każdy artykuł zaczyna się od własnego
wiersza "Art. N." — gotowy do chunkowania per artykuł.

Dlaczego PDF, a nie /text.html:
    Endpoint /text.html aktu bazowego serwuje tekst OGŁOSZONY (pierwotny), np. KC
    art. 33 wciąż z brzmieniem z 1964 r., a KPA w starej numeracji 1–196. Cytowanie
    takiego tekstu w narzędziu prawnym byłoby błędem merytorycznym. Aktualny tekst
    jednolity ISAP udostępnia wyłącznie jako PDF (typ "U").

Numery z indeksem górnym (np. art. 22^1) są odtwarzane z metadanych spanów PDF
(mniejszy rozmiar czcionki / flaga superscript), dzięki czemu art. 22^1 nie zlewa
się z art. 221.
"""
import re
import time
import sys
from pathlib import Path

import requests
import fitz  # pymupdf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import KODEKSY, RAW_DIR  # noqa: E402

API_BASE = "https://api.sejm.gov.pl/eli/acts"
HEADERS = {"User-Agent": "legal-rag-mvp/1.0 (kancelaria; research)"}
TIMEOUT = 60

BODY_MIN_SIZE = 11.5     # treść aktu (czcionka ~12 pt)
SUPERSCRIPT_MAX = 8.6    # indeks górny numeru (czcionka ~8 pt)

# Standalone-owe nagłówki jednostek systematyzujących (do pominięcia).
_STRUCT = re.compile(
    r"^(KSIĘGA|CZĘŚĆ|TYTUŁ|DZIAŁ|ODDZIAŁ|ROZDZIAŁ|Księga|Część|Tytuł|Dział|Oddział|Rozdział)\b"
)
_ARTICLE_START = re.compile(r"^Art\.\s")
_ARTICLE_HEADER = re.compile(r"^Art\.\s*\d+(?:\^\d+)?[a-z]?\.\s*")
_FOOTNOTE_REF = re.compile(r"^\d+[a-z]?\)$")            # przypis, np. "1)" — pomijamy
_EMPTY_BODY = re.compile(r"^\(?(uchylony|utracił moc|skreślony|pominięty)\)?\.?\s*$", re.I)


def _get(url: str, as_bytes: bool = False):
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.content if as_bytes else resp


def fetch_act_meta(address: str) -> dict:
    data = _get(f"{API_BASE}/{address}").json()
    last_updated = _resolve_unified_date(data) or (data.get("changeDate") or "")[:10]
    pdf_url = _consolidated_pdf_url(address, data)
    return {
        "title": data.get("title", ""),
        "status": data.get("status", ""),
        "last_updated": last_updated,
        "pdf_url": pdf_url,
    }


def _consolidated_pdf_url(address: str, data: dict) -> str:
    """URL PDF tekstu ujednoliconego (typ 'U'); fallback na /text.pdf."""
    for t in data.get("texts", []):
        if t.get("type") == "U" and t.get("fileName"):
            return f"{API_BASE}/{address}/text/U/{t['fileName']}"
    return f"{API_BASE}/{address}/text.pdf"


def _resolve_unified_date(act_data: dict) -> str | None:
    refs = (act_data.get("references") or {}).get("Inf. o tekście jednolitym") or []
    if not refs or not refs[0].get("id"):
        return None
    try:
        time.sleep(0.5)
        meta = _get(f"{API_BASE}/{refs[0]['id']}").json()
    except requests.RequestException:
        return None
    return (meta.get("promulgation") or meta.get("announcementDate") or "")[:10] or None


def _render_line(spans: list[dict]) -> str | None:
    """Buduje tekst wiersza z listy spanów PDF lub None, gdy wiersz to przypis/stopka.

    Gating na poziomie wiersza: jeśli największa czcionka < BODY_MIN_SIZE, wiersz jest
    stopką/przypisem/nagłówkiem strony -> pomijamy w całości. W wierszu treści zostają
    spany treści; spany indeksu górnego (mniejsza czcionka) dołączamy jako '^N',
    z wyjątkiem markerów przypisów typu 'N)'.
    """
    if not spans:
        return None
    if max(s["size"] for s in spans) < BODY_MIN_SIZE:
        return None

    out = []
    for s in spans:
        text = s["text"].replace("\xa0", " ").replace("‑", "-")
        if not text.strip() and not out:
            continue
        if s["size"] <= SUPERSCRIPT_MAX:
            token = text.strip()
            if not token or _FOOTNOTE_REF.match(token):
                continue  # przypis -> pomijamy
            joiner = "^" if (out and out[-1][-1:].isalnum()) else ""
            out.append(joiner + token)
        else:
            out.append(text)
    line = re.sub(r"\s+", " ", "".join(out)).strip()
    return line or None


def _pdf_to_lines(pdf_bytes: bytes) -> list[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    lines: list[str] = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                rendered = _render_line(line["spans"])
                if rendered:
                    lines.append(rendered)
    doc.close()
    return lines


def lines_to_articles(lines: list[str]) -> str:
    """Łączy wiersze w artykuły; pomija nagłówki jednostek i artykuły uchylone."""
    articles: list[str] = []
    current: list[str] | None = None
    expect_title = False

    def flush():
        nonlocal current
        if not current:
            return
        text = re.sub(r"\s+", " ", " ".join(current)).strip()
        body = _ARTICLE_HEADER.sub("", text).strip()
        if text and body and not _EMPTY_BODY.match(body):
            articles.append(text)
        current = None

    for line in lines:
        if _STRUCT.match(line):
            expect_title = True
            continue
        if _ARTICLE_START.match(line):
            expect_title = False
            flush()
            current = [line]
            continue
        if expect_title:
            expect_title = False  # opisowy tytuł jednostki tuż po nagłówku -> pomijamy
            continue
        if current is not None:
            current.append(line)
    flush()
    return "\n\n".join(articles)


def fetch_from_isap(skrot: str) -> tuple[str, str]:
    """Pobiera tekst jednolity kodeksu, zapisuje do data/raw/{skrot}.txt.

    Zwraca (plain_text, last_updated). Rzuca wyjątek przy błędzie HTTP/parsowania.
    """
    if skrot not in KODEKSY:
        raise ValueError(f"Nieznany skrót kodeksu: {skrot}")
    address = KODEKSY[skrot]["address"]

    meta = fetch_act_meta(address)
    time.sleep(0.5)
    pdf_bytes = _get(meta["pdf_url"], as_bytes=True)
    plain = lines_to_articles(_pdf_to_lines(pdf_bytes))
    if not plain.strip():
        raise ValueError(f"[{skrot}] Pusty tekst po parsowaniu PDF z {meta['pdf_url']}.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{skrot}.txt").write_text(plain, encoding="utf-8")
    return plain, meta["last_updated"]
