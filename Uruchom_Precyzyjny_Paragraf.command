#!/bin/bash
# Klikalny launcher aplikacji "Precyzyjny Paragraf".
# Dwuklik w Finderze otwiera Terminal i uruchamia aplikacje w przegladarce.

APP_DIR="/Users/konraddrozdz/PrecyzyjnyParagraf/legal_rag"
cd "$APP_DIR" || { echo "Nie znaleziono folderu aplikacji: $APP_DIR"; read -r; exit 1; }

echo "==================================================="
echo "   §  Precyzyjny Paragraf — uruchamianie..."
echo "==================================================="
echo

if [ ! -f ".env" ]; then
  echo "BLAD: brak pliku .env z kluczem Google."
  echo
  echo "Utworz go raz, wpisujac w Terminalu (podmien AIza... na swoj klucz):"
  echo "  cd \"$APP_DIR\" && echo \"GOOGLE_API_KEY=AIza...\" > .env"
  echo
  echo "Klucz zdobedziesz za darmo na https://aistudio.google.com (Get API key)."
  echo
  echo "Nacisnij Enter, aby zamknac to okno."
  read -r
  exit 1
fi

echo "Za chwile otworzy sie przegladarka z aplikacja."
echo "Aby zatrzymac aplikacje: wroc do tego okna i nacisnij Ctrl + C."
echo

exec "$APP_DIR/.venv/bin/streamlit" run "$APP_DIR/app.py"
