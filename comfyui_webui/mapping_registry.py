"""Workflow mapping registry for comfyui_webui.

A mapping links a workflow template to a named preset of generation parameters.
Mappings are stored in ``data/mappings.json``.  Each record contains:

- ``name``          – unique slug identifier
- ``display_name``  – user-facing label (e.g. "KI Schnell")
- ``template_name`` – name of the linked workflow template (from template_registry)
- ``checkpoint``    – default model checkpoint filename
- ``ollama_model``  – default Ollama model name
- ``steps``         – default step count
- ``cfg``           – default CFG scale
- ``seed``          – default seed (-1 = random)
- ``width``         – default image width
- ``height``        – default image height
- ``sampler``       – default sampler name
- ``scheduler``     – default scheduler name
- ``image_count``   – default number of images
- ``enabled``       – bool; hidden from users when False
- ``created_at``    – ISO timestamp
- ``updated_at``    – ISO timestamp
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
MAPPINGS_FILE = DATA_DIR / "mappings.json"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_mappings() -> list[dict[str, Any]]:
    if not MAPPINGS_FILE.exists():
        return []
    try:
        data = json.loads(MAPPINGS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("mapping_registry: could not load mappings: %s", exc)
    return []


def save_mappings(mappings: list[dict[str, Any]]) -> None:
    _ensure_data_dir()
    MAPPINGS_FILE.write_text(
        json.dumps(mappings, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def get_enabled_mappings() -> list[dict[str, Any]]:
    """Return mappings that are enabled (shown to all authenticated users)."""
    return [m for m in load_mappings() if m.get("enabled", True)]


def get_all_mappings() -> list[dict[str, Any]]:
    return load_mappings()


def get_mapping(name: str) -> dict[str, Any] | None:
    for m in load_mappings():
        if m.get("name") == name:
            return m
    return None


def register_mapping(
    name: str,
    display_name: str,
    template_name: str = "default",
    checkpoint: str = "",
    ollama_model: str = "",
    steps: int = 30,
    cfg: float = 7.0,
    seed: int = -1,
    width: int = 1024,
    height: int = 1024,
    sampler: str = "euler",
    scheduler: str = "normal",
    image_count: int = 1,
    enabled: bool = True,
) -> dict[str, Any]:
    """Register a new mapping or fully replace an existing one."""
    mappings = load_mappings()
    now = datetime.datetime.utcnow().isoformat()
    for m in mappings:
        if m["name"] == name:
            m.update(
                {
                    "display_name": display_name,
                    "template_name": template_name,
                    "checkpoint": checkpoint,
                    "ollama_model": ollama_model,
                    "steps": steps,
                    "cfg": cfg,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "sampler": sampler,
                    "scheduler": scheduler,
                    "image_count": image_count,
                    "enabled": enabled,
                    "updated_at": now,
                }
            )
            save_mappings(mappings)
            return m

    record: dict[str, Any] = {
        "name": name,
        "display_name": display_name,
        "template_name": template_name,
        "checkpoint": checkpoint,
        "ollama_model": ollama_model,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "width": width,
        "height": height,
        "sampler": sampler,
        "scheduler": scheduler,
        "image_count": image_count,
        "enabled": enabled,
        "created_at": now,
        "updated_at": now,
    }
    mappings.append(record)
    save_mappings(mappings)
    return record


def update_mapping(name: str, **fields: Any) -> dict[str, Any] | None:
    """Update allowed fields for a mapping record."""
    allowed = {
        "display_name",
        "template_name",
        "checkpoint",
        "ollama_model",
        "steps",
        "cfg",
        "seed",
        "width",
        "height",
        "sampler",
        "scheduler",
        "image_count",
        "enabled",
    }
    mappings = load_mappings()
    now = datetime.datetime.utcnow().isoformat()
    for m in mappings:
        if m["name"] == name:
            for key, value in fields.items():
                if key in allowed:
                    m[key] = value
            m["updated_at"] = now
            save_mappings(mappings)
            return m
    return None


def delete_mapping(name: str) -> bool:
    mappings = load_mappings()
    new = [m for m in mappings if m["name"] != name]
    if len(new) == len(mappings):
        return False
    save_mappings(new)
    return True
