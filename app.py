"""Streamlit UI — Precyzyjny Paragraf: asystent prawny RAG."""
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from config import KODEKSY  # noqa: E402
from llm.prompts import build_messages  # noqa: E402
from llm.gemini_client import ask_gemini_stream  # noqa: E402
from ingest.document_loader import extract_text, SUPPORTED  # noqa: E402

st.set_page_config(page_title="Precyzyjny Paragraf", page_icon="§", layout="wide")

CSS = """
<style>
:root { --navy:#16233d; --navy2:#213358; --accent:#b08d57; --ink:#1c2433; --muted:#5d6b80; }
.block-container { max-width: 1080px; padding-top: 2.2rem; }
.pp-hero { border-left: 5px solid var(--accent); padding: 0.2rem 0 0.2rem 1.1rem; margin-bottom: 0.4rem; }
.pp-hero h1 { font-size: 2.1rem; margin: 0; color: var(--navy); letter-spacing: -0.5px; }
.pp-hero p { font-size: 1.05rem; color: var(--muted); margin: 0.35rem 0 0; }
.pp-card { background:#fff; border:1px solid #e6e9ef; border-left:3px solid var(--accent);
  border-radius:10px; padding:0.95rem 1.05rem; height:100%; }
.pp-card h4 { margin:0 0 0.35rem; font-size:0.98rem; color:var(--navy); }
.pp-card p { margin:0; font-size:0.86rem; color:var(--muted); line-height:1.4; }
.pp-answer { background:#fbfaf7; border:1px solid #eee4d4; border-radius:12px;
  padding:1.2rem 1.4rem; font-size:1.02rem; line-height:1.65; color:var(--ink); }
.pp-docchip { display:inline-block; background:#eef3ff; border:1px solid #cfe0ff;
  color:#2a4a8c; border-radius:20px; padding:0.2rem 0.7rem; font-size:0.82rem; }
.stButton>button { background:var(--navy); color:#fff; border:0; border-radius:8px;
  padding:0.55rem 1.4rem; font-weight:600; }
.stButton>button:hover { background:var(--navy2); color:#fff; }
</style>
"""

VALUE_CARDS = [
    ("Zero zmyślonych przepisów",
     "Odpowiada wyłącznie z zaindeksowanych aktów. Gdy brak podstawy, mówi wprost "
     "„nie znaleziono” — zamiast wymyślić numer artykułu, jak robi to zwykły chatbot."),
    ("Aktualny tekst jednolity",
     "Przepisy prosto z ISAP (sejm.gov.pl) w aktualnym brzmieniu, z datą stanu prawnego "
     "— nie z pamięci modelu sprzed lat."),
    ("Właściwa hierarchia źródeł",
     "Konstytucja → Kodeks → Ustawa → Rozporządzenie. Nie powoła rozporządzenia tam, "
     "gdzie tę samą kwestię reguluje kodeks."),
    ("Weryfikowalne cytaty",
     "Każdy powołany artykuł pokazany w oryginale — źródło sprawdzasz jednym kliknięciem, "
     "a styl odpowiedzi jest prawniczy, nie „chatbotowy”."),
]


@st.cache_resource(show_spinner="Ładowanie modelu i indeksów prawnych...")
def get_searcher():
    from retrieval.hybrid_search import load_searcher
    return load_searcher()


def index_stats(searcher) -> dict:
    arts, frags, dates = defaultdict(set), defaultdict(int), {}
    for c in searcher.bm25.chunks:
        s = c["source"]
        arts[s].add(c["article"])
        frags[s] += 1
        dates[s] = c.get("last_updated", "?")
    return {s: {"articles": len(arts[s]), "fragments": frags[s], "date": dates.get(s, "?")}
            for s in arts}


def render_value_section():
    cols = st.columns(len(VALUE_CARDS))
    for col, (title, body) in zip(cols, VALUE_CARDS):
        col.markdown(f"<div class='pp-card'><h4>{title}</h4><p>{body}</p></div>",
                     unsafe_allow_html=True)
    with st.expander("Porównanie z ChatGPT — dlaczego to narzędzie do pracy prawnika"):
        st.markdown(
            "| | Zwykły ChatGPT | Precyzyjny Paragraf |\n"
            "|---|---|---|\n"
            "| **Źródło** | Pamięć modelu | Zaindeksowane akty z ISAP |\n"
            "| **Halucynacje artykułów** | Częste | Wykluczone — cytuje tylko z bazy |\n"
            "| **Aktualność** | Nieznana data | Tekst jednolity z datą stanu prawnego |\n"
            "| **Hierarchia źródeł** | Przypadkowa | Egzekwowana (Konstytucja → Rozporządzenie) |\n"
            "| **Weryfikacja** | Brak | Podgląd oryginalnych fragmentów |\n"
            "| **Styl** | Wypunktowania, grzeczności | Ciągły język prawniczy |\n"
        )


def render_sidebar(stats: dict):
    st.sidebar.header("Baza aktów prawnych")
    if not stats:
        st.sidebar.warning("Baza pusta. Uruchom:\n\n```\npython ingest/build_index.py\n```")
    else:
        for skrot, meta in KODEKSY.items():
            s = stats.get(skrot)
            if s:
                st.sidebar.markdown(
                    f"**{skrot}** · {meta['full']}  \n"
                    f"<span style='color:#5d6b80;font-size:0.82rem'>"
                    f"{s['articles']} art. · stan na {s['date']}</span>",
                    unsafe_allow_html=True,
                )
    st.sidebar.divider()
    st.sidebar.caption(
        f"Zapytań w tej sesji: **{st.session_state.get('query_count', 0)}** "
        "(limit darmowego tieru Google: 250/dzień)"
    )
    st.sidebar.caption(
        "Uwaga: na darmowym tierze Google może wykorzystywać zapytania do trenowania "
        "modeli. Do dokumentów z prawdziwymi danymi klienta wymagany płatny tier + "
        "anonimizacja."
    )


def render_sources(chunks: list[dict]):
    srcs = ", ".join(sorted({f"{c['source']} art. {c['article']}" for c in chunks}))
    st.caption(f"Wykorzystano {len(chunks)} fragment(ów): {srcs}")
    with st.expander("Fragmenty użyte jako kontekst (do weryfikacji)"):
        for i, c in enumerate(chunks, 1):
            st.markdown(
                f"**Fragment {i} — {c['source_full']} ({c['source']}), art. {c['article']}** "
                f"· stan na {c['last_updated']}"
            )
            st.write(c["text"])
            if i < len(chunks):
                st.divider()


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("query_count", 0)

    st.markdown(
        "<div class='pp-hero'><h1>§ Precyzyjny Paragraf</h1>"
        "<p>Asystent prawny oparty na polskich aktach prawnych — odpowiedzi z dokładnym "
        "cytowaniem, bez zmyślonych przepisów.</p></div>",
        unsafe_allow_html=True,
    )
    render_value_section()
    st.divider()

    if not os.environ.get("GOOGLE_API_KEY"):
        st.error(
            "Brak klucza **GOOGLE_API_KEY**. Uzyskaj go bezpłatnie na "
            "[aistudio.google.com](https://aistudio.google.com) → *Get API key*, "
            "i wpisz do pliku `.env`:\n\n```\nGOOGLE_API_KEY=AIza...\n```"
        )
        st.stop()

    try:
        searcher = get_searcher()
    except Exception as e:
        st.error(f"Nie udało się załadować indeksów: {e}")
        st.stop()

    stats = index_stats(searcher)
    render_sidebar(stats)
    if not stats:
        st.warning("Indeks jest pusty — uruchom `python ingest/build_index.py` przed zapytaniem.")

    query = st.text_area(
        "Zadaj pytanie prawne",
        height=120,
        placeholder="np. Kiedy dłużnik odpowiada za niewykonanie zobowiązania?",
    )
    uploaded = st.file_uploader(
        "Opcjonalnie: załącz dokument do analizy (PDF / DOCX / TXT)",
        type=[ext.lstrip(".") for ext in SUPPORTED],
    )
    go = st.button("Szukaj w prawie", type="primary", disabled=not stats)

    if go and query.strip():
        document_text, document_name = None, None
        if uploaded is not None:
            try:
                document_text, truncated = extract_text(uploaded.getvalue(), uploaded.name)
                document_name = uploaded.name
                if not document_text.strip():
                    st.warning(f"Nie udało się odczytać tekstu z „{uploaded.name}”.")
                    document_text = None
                elif truncated:
                    st.info(f"Dokument „{uploaded.name}” jest długi — do analizy użyto "
                            "początkowego fragmentu.")
            except Exception as e:
                st.error(f"Błąd odczytu dokumentu: {e}")

        chunks = searcher.search(query.strip())
        system_instruction, user_content = build_messages(
            query.strip(), chunks, document_text, document_name)

        st.subheader("Odpowiedź")
        if document_name:
            st.markdown(f"<span class='pp-docchip'>Analiza z dokumentem: {document_name}</span>",
                        unsafe_allow_html=True)
        try:
            with st.container():
                answer = st.write_stream(ask_gemini_stream(system_instruction, user_content))
        except Exception as e:
            st.error(f"Błąd wywołania Gemini: {e}")
            return

        st.session_state.query_count += 1
        st.session_state["last"] = {"answer": answer, "chunks": chunks, "doc": document_name}
        if chunks:
            render_sources(chunks)
        else:
            st.caption("Wyszukiwanie nie zwróciło pasujących fragmentów.")

    elif st.session_state.get("last"):
        last = st.session_state["last"]
        st.subheader("Odpowiedź")
        if last.get("doc"):
            st.markdown(f"<span class='pp-docchip'>Analiza z dokumentem: {last['doc']}</span>",
                        unsafe_allow_html=True)
        st.markdown(f"<div class='pp-answer'>{last['answer']}</div>", unsafe_allow_html=True)
        if last["chunks"]:
            render_sources(last["chunks"])


if __name__ == "__main__":
    main()
