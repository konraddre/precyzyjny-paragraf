# § Precyzyjny Paragraf

Asystent prawny RAG dla polskiego prawa. Odpowiada **wyłącznie** na podstawie
zaindeksowanych aktów prawnych z ISAP, z dokładnym cytowaniem artykułów — bez
zmyślonych przepisów typowych dla zwykłych chatbotów.

Zbudowany jako narzędzie pracy dla prawnika: wpisujesz pytanie (opcjonalnie
załączasz dokument), a w odpowiedzi dostajesz wywód w języku prawniczym z
powołaniem konkretnych przepisów i podglądem ich oryginalnego brzmienia do
weryfikacji.

## Dlaczego nie zwykły ChatGPT

| | Zwykły ChatGPT | Precyzyjny Paragraf |
|---|---|---|
| **Źródło** | Pamięć modelu | Zaindeksowane akty z ISAP |
| **Halucynacje artykułów** | Częste | Wykluczone — cytuje tylko z bazy |
| **Aktualność** | Nieznana data | Tekst jednolity z datą stanu prawnego |
| **Hierarchia źródeł** | Przypadkowa | Egzekwowana (Konstytucja → Rozporządzenie) |
| **Weryfikacja** | Brak | Podgląd oryginalnych fragmentów |
| **Styl** | Wypunktowania, grzeczności | Ciągły język prawniczy |

## Jak to działa

```
Pytanie
  │
  ├─► Wyszukiwanie hybrydowe
  │     ├─ BM25 (słowa kluczowe, tokenizacja PL)
  │     ├─ Semantyczne (multilingual-e5-base, Qdrant)
  │     └─ Fuzja RRF → top-5 fragmentów
  │
  ├─► Prompt antyhalucynacyjny + fragmenty (+ opcjonalny dokument)
  │
  └─► Gemini 2.5 Flash (streaming) ─► Odpowiedź z cytatami + źródła do wglądu
```

Baza obejmuje 7 kodeksów (tekst jednolity z ISAP): **KC, KPC, KK, KPK, KP, KPA,
KSH**. Aktualne liczby artykułów i daty stanu prawnego widać na żywo w panelu
bocznym aplikacji.

## Stack

- **UI:** Streamlit
- **LLM:** Google Gemini 2.5 Flash
- **Embeddingi:** `intfloat/multilingual-e5-base` (sentence-transformers)
- **Wektorowa baza:** Qdrant (tryb lokalny, bez serwera)
- **Wyszukiwanie leksykalne:** BM25 (`rank_bm25`)
- **Źródło prawa:** ISAP / API ELI (`api.sejm.gov.pl`)
- **Dokumenty:** PyMuPDF (PDF), python-docx (DOCX)

## Struktura projektu

```
app.py                  # Streamlit UI
config.py               # wspólne stałe, rejestr kodeksów
requirements.txt
ingest/
  isap_client.py        # pobieranie + parsowanie tekstu jednolitego (PDF) z ISAP
  chunker.py            # podział na artykuły (i paragrafy dla długich)
  document_loader.py    # ekstrakcja tekstu z PDF/DOCX/TXT
  build_index.py        # skrypt ingestion (budowa/aktualizacja indeksu)
retrieval/
  embedder.py           # model embeddingów
  bm25_index.py         # indeks BM25 + tokenizacja PL
  vector_store.py       # operacje na Qdrant
  hybrid_search.py      # fuzja RRF
llm/
  prompts.py            # prompt systemowy + budowa wiadomości
  gemini_client.py      # streaming z Gemini
data/                   # zaindeksowana baza (Qdrant + BM25 + surowe teksty)
.github/workflows/
  refresh-index.yml     # cotygodniowa automatyczna aktualizacja bazy z ISAP
```

## Uruchomienie lokalne

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# klucz Google (darmowy: https://aistudio.google.com → Get API key)
echo "GOOGLE_API_KEY=AIza..." > .env

# zbuduj indeks (pobiera teksty z ISAP, liczy embeddingi)
python ingest/build_index.py

streamlit run app.py
```

`build_index.py` przy kolejnych uruchomieniach pomija już zaindeksowane kodeksy;
`--force` przebudowuje wszystko od zera, `--force KC KPC` tylko wskazane.

## Wdrożenie (Streamlit Community Cloud)

1. Repo na GitHubie (indeks `data/` jest commitowany, więc aplikacja działa od razu).
2. Na [share.streamlit.io](https://share.streamlit.io) → **Create app**, wskaż repo,
   gałąź `main`, główny plik `app.py`.
3. W **Advanced settings → Secrets** wklej klucz (nie trafia do repo):
   ```toml
   GOOGLE_API_KEY = "AIza..."
   ```
4. (Zalecane) **Settings → Sharing** → ogranicz dostęp do wybranych e-maili, by
   nikt postronny nie zużywał limitu zapytań.

## Automatyczna aktualizacja bazy

Workflow `.github/workflows/refresh-index.yml` uruchamia się **co tydzień**
(poniedziałek 03:00 UTC) oraz ręcznie z zakładki *Actions*. Pobiera świeże teksty
jednolite z ISAP, przebudowuje indeks i wypycha zmiany — Streamlit wykrywa nowy
commit i sam przeładowuje aplikację z aktualnym stanem prawnym. Jeśli pobranie
któregoś kodeksu się nie powiedzie, poprzednia wersja bazy jest zachowywana.

## Ograniczenia i zastrzeżenia

- Odpowiedzi mają **charakter informacyjny i nie stanowią porady prawnej**.
- Baza to **zamrożony stan prawny** z daty ostatniego indeksowania (widocznej w
  aplikacji) — nowelizacje po tej dacie nie są uwzględniane do następnego odświeżenia.
- Na **darmowym tierze Google** zapytania mogą być wykorzystywane do trenowania
  modeli. Do dokumentów z prawdziwymi danymi klienta wymagany jest płatny tier
  oraz anonimizacja (poza zakresem tego MVP).
