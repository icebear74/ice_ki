"""Shared application paths."""
from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("COMFYUI_WEBUI_DATA_DIR") or APP_DIR / "data").expanduser().resolve()
