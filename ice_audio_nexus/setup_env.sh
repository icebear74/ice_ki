#!/bin/bash
# setup_env.sh – Richtet die Python-Umgebung für ice_audio_nexus ein.
# Nutzt Python 3.12 (bevorzugt) oder 3.11.
# Führe dieses Skript im Verzeichnis ice_audio_nexus aus.

set -e

echo "🚀 Starte Setup für ice_audio_nexus ..."

# Python-Interpreter ermitteln
PYTHON_EXE=$(command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null || command -v python3 2>/dev/null)
if [ -z "$PYTHON_EXE" ]; then
    echo "❌ Kein Python 3.11/3.12 gefunden. Bitte installieren."
    exit 1
fi
echo "✅ Python-Interpreter: $PYTHON_EXE"

# Virtual Environment erstellen
if [ ! -d "venv" ]; then
    "$PYTHON_EXE" -m venv venv
    echo "✅ venv erstellt."
else
    echo "ℹ️  venv existiert bereits – überspringe Erstellung."
fi

# Aktivieren
source venv/bin/activate

# pip upgraden
pip install --upgrade pip --quiet

echo "📦 Installiere Kern-Abhängigkeiten ..."

# MariaDB-Connector (benötigt libmariadb-dev auf dem System)
pip install mariadb python-dotenv --quiet

# PyTorch mit CUDA 12.x (für Tesla P4 / P100)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet

# Audio-Processing & Diarization
pip install pyannote.audio faster-whisper librosa soundfile --quiet

# Web-Interface
pip install "fastapi[standard]" uvicorn[standard] jinja2 python-multipart aiofiles --quiet

echo ""
echo "✅ Setup abgeschlossen!"
echo ""
echo "Nächste Schritte:"
echo "  1. Kopiere .env.example → .env und trage deine DB-Zugangsdaten ein."
echo "  2. Aktiviere die Umgebung:  source venv/bin/activate"
echo "  3. Starte die Web-GUI:      uvicorn web_ui.api:app --reload --host 0.0.0.0 --port 8765"
echo "  4. Scanner starten:         python -m processor.scanner --video /pfad/zum/video.mkv"
