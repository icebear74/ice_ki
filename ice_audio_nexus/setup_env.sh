#!/bin/bash
set -e

echo "🚀 Fix für huggingface_hub Inkompatibilität..."

# venv aktivieren
source venv/bin/activate

# 1. huggingface_hub auf eine Version bringen, die noch use_auth_token versteht
pip install "huggingface_hub<0.25.0"

# 2. Kurzer Check
python3 << END
from huggingface_hub import hf_hub_download
import pyannote.audio
print(f"huggingface_hub Version: {importlib_metadata.version('huggingface_hub') if 'importlib_metadata' in locals() else 'Check manual'}")
print("✅ huggingface_hub Downgrade abgeschlossen.")
END

echo "✅ Fertig. Jetzt sollte Pipeline.from_pretrained() endlich durchlaufen."
