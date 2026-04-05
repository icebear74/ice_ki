#!/bin/bash
# =============================================================================
# ice_audio_nexus – Setup Script
# =============================================================================
# Stable production setup for Tesla P100 (SM 6.0) / P4 (SM 6.1) Pascal GPUs.
#
# Key version pins (derived from debugging session):
#   Python:          3.12 (preferred) or 3.11
#   torch:           2.4.1+cu118  (Pascal support; cu130 breaks CUBLAS)
#   torchaudio:      2.4.1+cu118
#   numpy:           <2.0.0       (avoid AttributeError: np.NaN)
#   huggingface_hub: <0.25.0      (keep use_auth_token param for pyannote 3.1.1)
#                    ⚠ MUSS nach wespeaker/s3prl erneut gepinnt werden –
#                    s3prl → transformers 5.x zieht eine neuere Version nach!
#   transformers:    <5.0.0       (transformers 5.x braucht hub>=1.5, inkompatibel)
#                    ⚠ MUSS ebenfalls nach s3prl erneut gepinnt werden!
#   pyannote.audio:  ==3.1.1      (stable for numpy<2 + old PyTorch)
#   s3prl:           (wespeaker-Abhängigkeit, nicht automatisch mitinstalliert)
#
# Usage:
#   cd ice_audio_nexus
#   bash setup_env.sh
# =============================================================================

set -e

BOLD='\033[1m'
GREEN='\033[92m'
CYAN='\033[96m'
YELLOW='\033[93m'
RED='\033[91m'
RESET='\033[0m'

echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   ice_audio_nexus – Production Setup (Pascal P100/P4)       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ---------------------------------------------------------------------------
# 1. Resolve Python binary (3.12 preferred, 3.11 fallback)
# ---------------------------------------------------------------------------
echo -e "${CYAN}🐍 Schritt 1: Python-Executable ermitteln...${RESET}"
PY_BIN=""
for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        PY_VER=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [ "$MAJOR" = "3" ] && [ "$MINOR" -ge 11 ]; then
            PY_BIN="$candidate"
            echo -e "${GREEN}✓ Verwende $candidate (Version $PY_VER)${RESET}"
            break
        fi
    fi
done

if [ -z "$PY_BIN" ]; then
    echo -e "${RED}❌ Python 3.11+ nicht gefunden. Bitte installieren.${RESET}"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Create / activate venv
# ---------------------------------------------------------------------------
echo -e "\n${CYAN}📦 Schritt 2: Virtuelle Umgebung erstellen...${RESET}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠ Vorhandene venv wird gelöscht und neu erstellt...${RESET}"
    rm -rf venv
fi
"$PY_BIN" -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ venv bereit: $(python --version)${RESET}"

# ---------------------------------------------------------------------------
# 3. Install all high-level AI packages FIRST
#    (they will pull in a wrong/recent torch – we fix that in step 5)
# ---------------------------------------------------------------------------
echo -e "\n${CYAN}🤖 Schritt 3: KI-Pakete installieren (torch kommt danach)...${RESET}"

# numpy<2 must be pinned before anything else pulls in 2.x
pip install "numpy<2.0.0"

# huggingface_hub<0.25 keeps use_auth_token support for pyannote 3.1.1
pip install "huggingface_hub<0.25.0"

# pyannote.audio 3.1.1 – stable on numpy<2 + old torch
pip install "pyannote.audio==3.1.1"

# faster-whisper + audio helpers
pip install faster-whisper librosa soundfile audioread

# matplotlib – required by pyannote.audio internally (tasks/segmentation/mixins.py)
pip install matplotlib

# Remove torchcodec – it requires CUDA 12.x+ and breaks Pascal cards
pip uninstall -y torchcodec 2>/dev/null || true

echo -e "${GREEN}✓ KI-Pakete installiert${RESET}"

# ---------------------------------------------------------------------------
# 4. Web-UI & DB packages + image processing + audio enhancement
# ---------------------------------------------------------------------------
echo -e "\n${CYAN}>> Schritt 4: Web-UI-Pakete installieren...${RESET}"
pip install \
    "fastapi[standard]" \
    "uvicorn[standard]" \
    jinja2 \
    python-multipart \
    aiofiles \
    "python-dotenv" \
    mariadb \
    "Pillow>=10.0.0"

# deepfilternet – audio noise suppression (Pascal GPU / CUDA 11.8 compatible)
# Uses the PyTorch version installed in step 5, so install before the torch pin.
# deepfilterlib (dependency of deepfilternet) is written in Rust and requires Cargo
# when no pre-built wheel is available. Install Rust automatically via rustup if missing.
echo -e "${CYAN}  → Prüfe Rust/Cargo für deepfilternet...${RESET}"
if ! command -v cargo &>/dev/null; then
    echo -e "${YELLOW}  ⚠ Rust/Cargo nicht gefunden – installiere via rustup...${RESET}"
    if curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path; then
        # shellcheck disable=SC1090
        source "$HOME/.cargo/env"
        echo -e "${GREEN}  ✓ Rust $(rustc --version 2>/dev/null | awk '{print $2}') installiert${RESET}"
    else
        echo -e "${RED}  ✗ Rust-Installation fehlgeschlagen. deepfilternet wird übersprungen.${RESET}"
        echo -e "${YELLOW}    Manuell installieren: https://rustup.rs${RESET}"
    fi
else
    echo -e "${GREEN}  ✓ Rust/Cargo bereits vorhanden ($(cargo --version 2>/dev/null))${RESET}"
fi

echo -e "${CYAN}  → Installiere deepfilternet...${RESET}"
if pip install deepfilternet; then
    echo -e "${GREEN}  ✓ deepfilternet installiert${RESET}"
else
    echo -e "${YELLOW}  ⚠ deepfilternet konnte nicht installiert werden.${RESET}"
    echo -e "${YELLOW}    Audio-Rauschunterdrückung wird deaktiviert.${RESET}"
fi

# wespeaker – Speaker-Embedding-Modell (nicht auf PyPI – Installation über GitHub)
# Offizielle Quelle: https://github.com/wenet-e2e/wespeaker
echo -e "${CYAN}  → Installiere wespeaker (von GitHub)...${RESET}"
if pip install git+https://github.com/wenet-e2e/wespeaker.git; then
    echo -e "${GREEN}  ✓ wespeaker installiert${RESET}"
else
    echo -e "${YELLOW}  ⚠ wespeaker konnte nicht installiert werden.${RESET}"
    echo -e "${YELLOW}    Speaker-Embedding-Extraktion wird deaktiviert (Fallback auf leere Vektoren).${RESET}"
    echo -e "${YELLOW}    Manuell installieren: pip install git+https://github.com/wenet-e2e/wespeaker.git${RESET}"
fi

# s3prl – Pflichtabhängigkeit von wespeaker (wird nicht automatisch mitinstalliert)
# Ohne s3prl schlägt wespeaker.load_model() mit "No module named 's3prl'" fehl.
echo -e "${CYAN}  → Installiere s3prl (wespeaker-Abhängigkeit)...${RESET}"
if pip install s3prl; then
    echo -e "${GREEN}  ✓ s3prl installiert${RESET}"
else
    echo -e "${YELLOW}  ⚠ s3prl konnte nicht installiert werden – wespeaker wird möglicherweise fehlschlagen.${RESET}"
fi

# openai-whisper – transitiv benötigt von s3prl (s3prl.upstream.whisper wird beim Import geladen)
# s3prl importiert ALLE Upstream-Module beim Start (eager). Der Whisper-Upstream macht kein
# try/except um 'import whisper', weshalb ohne openai-whisper ein ImportError auftritt der
# wespeaker bricht – auch wenn wir für Transkription faster-whisper nutzen.
# openai-whisper und faster-whisper koexistieren: Module heißen 'whisper' vs 'faster_whisper'.
echo -e "${CYAN}  → Installiere openai-whisper (s3prl-Abhängigkeit für whisper-Upstream)...${RESET}"
if pip install openai-whisper; then
    echo -e "${GREEN}  ✓ openai-whisper installiert${RESET}"
else
    echo -e "${YELLOW}  ⚠ openai-whisper konnte nicht installiert werden – wespeaker/s3prl wird fehlschlagen.${RESET}"
fi

# ⚠ KRITISCH: huggingface_hub, transformers und numpy nach wespeaker/s3prl erneut pinnen!
# Abhängigkeitskette: s3prl → transformers 5.x → huggingface_hub >= 1.5
# Beide Pins sind nötig:
#   huggingface_hub < 0.25.0  → pyannote.audio 3.1.1 braucht use_auth_token-Parameter
#   transformers    < 5.0.0   → transformers 5.x importiert is_offline_mode, das in hub 0.24.x
#                               nicht (mehr) auf Modulebene verfügbar ist
echo -e "${CYAN}  → Stelle huggingface_hub < 0.25.0, transformers < 5.0.0 und numpy < 2.0.0 sicher (Re-Pin nach s3prl)...${RESET}"
pip install "huggingface_hub<0.25.0" "transformers<5.0.0" "numpy<2.0.0"
echo -e "${GREEN}  ✓ huggingface_hub, transformers und numpy korrekt gepinnt${RESET}"

echo -e "${GREEN}✓ Web-UI-Pakete + Pillow + DeepFilterNet installiert${RESET}"

# ---------------------------------------------------------------------------
# 5. Force-install compatible torch (CUDA 11.8 – Pascal SM 6.0/6.1 support)
#    This MUST come last to override whatever pyannote/whisper pulled in.
# ---------------------------------------------------------------------------
echo -e "\n${CYAN}🔥 Schritt 5: PyTorch 2.4.1+cu118 (Pascal-kompatibel) erzwingen...${RESET}"
pip uninstall -y torch torchaudio torchvision 2>/dev/null || true
pip install --no-cache-dir \
    "torch==2.4.1+cu118" \
    "torchaudio==2.4.1+cu118" \
    --index-url https://download.pytorch.org/whl/cu118
echo -e "${GREEN}✓ PyTorch 2.4.1+cu118 installiert${RESET}"

# ---------------------------------------------------------------------------
# 6. Full smoke-test: import every key module + version pins + GPU matmul
#    Faster than a full pipeline run – catches dependency hell immediately.
# ---------------------------------------------------------------------------
echo -e "\n${CYAN}🔍 Schritt 6: Vollständiger Import-Smoke-Test...${RESET}"
python3 << 'PYCHECK'
import sys
import importlib
import importlib.metadata

CRITICAL_FAIL = False   # will sys.exit(1) at the end if True

def ok(label, detail=""):
    suffix = f"  ({detail})" if detail else ""
    print(f"  ✅ {label}{suffix}")

def warn(label, detail=""):
    suffix = f"  ({detail})" if detail else ""
    print(f"  ⚠️  {label}{suffix}")

def fail(label, detail=""):
    global CRITICAL_FAIL
    CRITICAL_FAIL = True
    suffix = f"\n       {detail}" if detail else ""
    print(f"  ❌ {label}{suffix}")

def pkg_ver(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

# ── Python ──────────────────────────────────────────────────────────────────
print(f"\n── Python / Basis ──────────────────────────────────────────────")
print(f"  Python {sys.version.split()[0]}")

# ── PyTorch ─────────────────────────────────────────────────────────────────
print(f"\n── PyTorch & CUDA ──────────────────────────────────────────────")
try:
    import torch
    ver = torch.__version__
    if "cu118" not in ver:
        fail("PyTorch", f"Erwartet cu118, gefunden: {ver}  →  pip install torch==2.4.1+cu118 ...")
    else:
        ok("PyTorch", ver)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name  = torch.cuda.get_device_name(i)
            maj, mn = torch.cuda.get_device_capability(i)
            mem  = torch.cuda.get_device_properties(i).total_memory // (1024*1024)
            ok(f"GPU {i}: {name}", f"SM {maj}.{mn}, {mem} MB")
        # MatMul smoke-test (war der CUBLAS-Absturzpunkt auf Pascal)
        try:
            a = torch.randn(64, 64, device="cuda")
            _ = a @ a
            torch.cuda.synchronize()
            ok("GPU MatMul (CUBLAS)")
        except Exception as e:
            fail("GPU MatMul", str(e))
    else:
        warn("CUDA nicht verfügbar – läuft auf CPU")
except ImportError as e:
    fail("PyTorch", str(e))

# ── NumPy ────────────────────────────────────────────────────────────────────
print(f"\n── NumPy ───────────────────────────────────────────────────────")
try:
    import numpy as np
    ver = np.__version__
    if tuple(int(x) for x in ver.split(".")[:2]) >= (2, 0):
        fail("NumPy", f"{ver} >= 2.0  →  pip install 'numpy<2.0.0'")
    else:
        ok("NumPy", ver)
except ImportError as e:
    fail("NumPy", str(e))

# ── huggingface_hub + transformers ──────────────────────────────────────────
print(f"\n── HuggingFace ─────────────────────────────────────────────────")
hf_ver = pkg_ver("huggingface_hub")
if hf_ver is None:
    fail("huggingface_hub", "nicht installiert")
else:
    parts = tuple(int(x) for x in hf_ver.split(".")[:2])
    if parts >= (0, 25):
        fail("huggingface_hub", f"{hf_ver} >= 0.25 bricht pyannote use_auth_token  →  pip install 'huggingface_hub<0.25.0'")
    else:
        ok("huggingface_hub", hf_ver)

tr_ver = pkg_ver("transformers")
if tr_ver is None:
    warn("transformers", "nicht installiert")
else:
    parts = tuple(int(x) for x in tr_ver.split(".")[:2])
    if parts >= (5, 0):
        fail("transformers", f"{tr_ver} >= 5.0 importiert is_offline_mode (fehlt in hub 0.24.x)  →  pip install 'transformers<5.0.0'")
    else:
        # Tatsächlich importieren – testet die hub-Kompatibilität live
        try:
            import transformers  # noqa: F401
            ok("transformers", tr_ver)
        except Exception as e:
            fail("transformers import", str(e))

# ── Audio I/O ────────────────────────────────────────────────────────────────
print(f"\n── Audio I/O ───────────────────────────────────────────────────")
for mod in ("soundfile", "librosa"):
    try:
        importlib.import_module(mod)
        ok(mod, pkg_ver(mod) or "?")
    except ImportError as e:
        fail(mod, str(e))

# ── faster-whisper (Transkription) ───────────────────────────────────────────
print(f"\n── Transkription ───────────────────────────────────────────────")
try:
    from faster_whisper import WhisperModel  # noqa: F401
    ok("faster-whisper", pkg_ver("faster-whisper") or "?")
except ImportError as e:
    fail("faster-whisper", str(e))

# ── pyannote.audio (Diarisierung) ────────────────────────────────────────────
print(f"\n── Diarisierung ────────────────────────────────────────────────")
try:
    from pyannote.audio import Pipeline  # noqa: F401
    ok("pyannote.audio", pkg_ver("pyannote.audio") or "?")
except ImportError as e:
    fail("pyannote.audio", str(e))

# ── DeepFilterNet (Rauschunterdrückung) ──────────────────────────────────────
print(f"\n── Rauschunterdrückung ─────────────────────────────────────────")
try:
    from df.enhance import enhance, init_df  # noqa: F401
    ok("deepfilternet", pkg_ver("deepfilternet") or "?")
except ImportError as e:
    warn("deepfilternet (optional)", str(e))

# ── WeSpeaker (Speaker-Embeddings) ───────────────────────────────────────────
print(f"\n── Speaker-Embeddings ──────────────────────────────────────────")
try:
    import wespeaker  # noqa: F401
    ok("wespeaker")
except ImportError as e:
    warn("wespeaker (optional)", str(e))

# ── MariaDB-Connector ─────────────────────────────────────────────────────────
print(f"\n── Datenbank ───────────────────────────────────────────────────")
try:
    import mariadb  # noqa: F401
    ok("mariadb", pkg_ver("mariadb") or "?")
except ImportError as e:
    fail("mariadb", str(e))

# ── Web-UI ────────────────────────────────────────────────────────────────────
print(f"\n── Web-UI ──────────────────────────────────────────────────────")
for mod, pkg in (("fastapi", "fastapi"), ("uvicorn", "uvicorn")):
    try:
        importlib.import_module(mod)
        ok(mod, pkg_ver(pkg) or "?")
    except ImportError as e:
        warn(f"{mod} (optional)", str(e))

# ── Ergebnis ──────────────────────────────────────────────────────────────────
print()
if CRITICAL_FAIL:
    print("❌ Mindestens ein kritischer Check fehlgeschlagen – bitte oben beheben!")
    sys.exit(1)
else:
    print("✅ Alle kritischen Checks bestanden. Scanner kann gestartet werden.")
PYCHECK

echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════════════╗"
echo "║   Setup abgeschlossen!                                      ║"
echo "║   Aktivieren:  source venv/bin/activate                    ║"
echo "║   Scanner:     python -m processor.scanner --help          ║"
echo "║   Web-UI:      uvicorn web_ui.api:app --host 0.0.0.0 ...   ║"
echo -e "╚══════════════════════════════════════════════════════════════╝${RESET}"
