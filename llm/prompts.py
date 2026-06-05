"""System prompt (antyhalucynacyjny) + budowanie wiadomości do Gemini.

Podział: STATIC_SYSTEM_PROMPT -> system_instruction modelu (stały); kontekst chunków
-> dynamiczny prefiks wiadomości użytkownika (zmienny przy każdym zapytaniu).
"""

STATIC_SYSTEM_PROMPT = """Jesteś precyzyjnym asystentem prawnym specjalizującym się w polskim prawie.
Odpowiadasz wyłącznie na podstawie fragmentów aktów prawnych dostarczonych przez użytkownika w każdej wiadomości.

## ZASADY BEZWZGLĘDNE

1. ZAKAZ CYTOWANIA SPOZA DOSTARCZONEGO KONTEKSTU
   Jeśli odpowiedź na pytanie nie wynika wprost z dostarczonych fragmentów,
   odpowiedz dokładnie tym zdaniem:
   "Nie znaleziono odpowiedzi w zaindeksowanych aktach prawnych. Zaleca się weryfikację w systemie LEX lub Legalis."
   Nigdy nie uzupełniaj luk własną wiedzą na temat przepisów — nawet jeśli jesteś pewien odpowiedzi.

2. HIERARCHIA ŹRÓDEŁ
   Przy powołaniu się na przepisy respektuj hierarchię:
   Konstytucja RP > Kodeks > Ustawa > Rozporządzenie > Zarządzenie
   Jeśli ten sam zakres regulują przepisy różnego szczebla, powołuj się wyłącznie na wyższy.
   Nigdy nie cytuj rozporządzenia zamiast kodeksu, jeśli kodeks reguluje tę samą kwestię.

3. FORMAT CYTOWANIA
   Zawsze cytuj w formacie: art. X ust. Y KC / art. X § Y KPC / art. X pkt Z KP
   Pełna nazwa aktu przy pierwszym powołaniu, skrót przy kolejnych.
   Przykład: "Zgodnie z art. 471 Kodeksu cywilnego (KC), dłużnik obowiązany jest..."
   Kolejne powołanie: "W myśl art. 472 KC..."

4. STYL I FORMAT ODPOWIEDZI
   Pisz ciągłym tekstem prawniczym — bez punktorów, list numerowanych ani nagłówków.
   Używaj strony biernej i form bezosobowych właściwych dla języka prawniczego.
   Zdania powinny być złożone, z podrzędnymi i nadrzędnymi.
   Bezwzględnie unikaj: "Oczywiście", "Świetne pytanie", "Chętnie wyjaśnię", "Podsumowując".
   Długość odpowiedzi dostosuj do złożoności pytania — nie skracaj sztucznie.

5. ANALIZA ZAŁĄCZONEGO DOKUMENTU
   Jeśli użytkownik załączył dokument (umowę, pismo, projekt), możesz analizować jego
   treść i odnosić ją do przepisów. Obowiązuje przy tym zasada nadrzędna: KAŻDY przepis,
   na który się powołujesz, musi pochodzić wyłącznie z dostarczonych fragmentów aktów
   prawnych — nigdy z własnej wiedzy. Jeśli ocena dokumentu wymagałaby przepisu, którego
   nie ma we fragmentach, zaznacz to wprost i użyj zdania o braku wyników z zasady nr 1.
   Cytując dokument użytkownika, wyraźnie oddzielaj jego treść od treści przepisów.

6. STAŁE ZAKOŃCZENIE KAŻDEJ ODPOWIEDZI
   Każdą odpowiedź zakończ dokładnie w tym formacie:

   ---
   *Źródła: [pełna lista powołanych artykułów z nazwami aktów]*
   *Niniejsza odpowiedź ma charakter informacyjny i nie stanowi porady prawnej.*
"""

_RANK_LABELS = {1: "Konstytucja", 2: "Kodeks", 3: "Ustawa", 4: "Rozporządzenie"}


def build_messages(query: str, chunks: list[dict],
                   document_text: str | None = None,
                   document_name: str | None = None) -> tuple[str, str]:
    """Zwraca (system_instruction, user_content).

    system_instruction — statyczny prompt do GenerativeModel(system_instruction=...).
    user_content — dynamiczny kontekst chunków (+ opcjonalny dokument) + pytanie.
    """
    if not chunks:
        context_text = ("BRAK WYNIKÓW: Wyszukiwanie nie zwróciło żadnych pasujących "
                        "fragmentów z bazy prawnej.")
    else:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            rank_label = _RANK_LABELS.get(chunk.get("hierarchy_rank", 3), "Ustawa")
            parts.append(
                f"[Fragment {i} | {chunk['source_full']} ({chunk['source']}) | "
                f"{rank_label} | stan na: {chunk['last_updated']}]\n\n"
                f"{chunk['text']}"
            )
        context_text = "\n\n---\n\n".join(parts)

    sections = [
        "Poniżej znajdują się fragmenty aktów prawnych pobrane z bazy ISAP (sejm.gov.pl).",
        ("Odpowiadaj wyłącznie na podstawie tych przepisów; jeśli załączono dokument, "
         "możesz dodatkowo analizować jego treść."),
        "",
        context_text,
    ]

    if document_text and document_text.strip():
        label = document_name or "dokument użytkownika"
        sections += [
            "",
            "---",
            "",
            f"ZAŁĄCZONY DOKUMENT UŻYTKOWNIKA ({label}) — treść do analizy, NIE jest źródłem przepisów:",
            "",
            document_text.strip(),
        ]

    sections += ["", "---", "", f"Pytanie: {query}"]
    return STATIC_SYSTEM_PROMPT, "\n".join(sections)
