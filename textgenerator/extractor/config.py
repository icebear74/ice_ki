"""Configuration of the character extraction service.

Every setting is read from the environment so the Kubernetes manifests stay
the single source of truth. Defaults are chosen so the service works inside
the ai-stack namespace without any additional configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_API_BASE_URL = "http://text-generation-webui.ai-stack.svc.cluster.local:5000/v1"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings of the extractor."""

    api_base_url: str
    model: str
    max_tokens: int
    request_timeout: float
    # Maximum context the backend was started with. Used to warn about
    # oversized inputs - it is NOT an application side hard truncation.
    context_size: int
    character_card_dir: Path
    profile_dir: Path
    raw_output_dir: Path
    allow_overwrite: bool
    # "off" (default, uncensored) or "sfw". Affects generated IMAGE prompts
    # only; the text extraction prompt never carries moderation wording.
    image_prompt_safety_mode: str

    @classmethod
    def from_env(cls) -> "Settings":
        safety_mode = os.environ.get("IMAGE_PROMPT_SAFETY_MODE", "off").strip().lower()
        if safety_mode not in {"off", "sfw"}:
            safety_mode = "off"
        return cls(
            api_base_url=os.environ.get("OOBABOOGA_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/"),
            model=os.environ.get("EXTRACTOR_MODEL", "").strip(),
            max_tokens=_env_int("EXTRACTOR_MAX_TOKENS", 2048),
            request_timeout=float(_env_int("EXTRACTOR_REQUEST_TIMEOUT", 600)),
            context_size=_env_int("TEXTGEN_CONTEXT_SIZE", 8192),
            character_card_dir=Path(os.environ.get("CHARACTER_CARD_DIR", "/data/characters")),
            profile_dir=Path(os.environ.get("PROFILE_DIR", "/data/extractor/profiles")),
            raw_output_dir=Path(os.environ.get("RAW_OUTPUT_DIR", "/data/extractor/raw")),
            allow_overwrite=_env_bool("EXTRACTOR_ALLOW_OVERWRITE", False),
            image_prompt_safety_mode=safety_mode,
        )
