"""Klient Google Gemini 2.5 Flash (streaming).

Darmowy tier Google AI Studio (aistudio.google.com): 250 zapytań/dzień, 10 RPM,
kontekst 1M tokenów. UWAGA: na darmowym tierze Google może używać zapytań do
trenowania modeli — do testów z danymi fikcyjnymi, nie z danymi klientów.
"""
import os

import google.generativeai as genai
from typing import Generator

MODEL_NAME = "gemini-2.5-flash"


def _require_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Brak GOOGLE_API_KEY. Uzyskaj klucz na https://aistudio.google.com "
            "('Get API key') i wpisz go do pliku .env."
        )
    return key


def ask_gemini_stream(system_instruction: str, user_content: str) -> Generator[str, None, None]:
    """Generator streamujący odpowiedź token po tokenie."""
    genai.configure(api_key=_require_api_key())
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_instruction,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=2048,
            temperature=0.1,  # niska temperatura = bardziej deterministyczne cytaty
        ),
    )
    response = model.generate_content(user_content, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text
