"""Template registry and approval store for comfyui_webui.

Templates are persisted in ``data/templates.json``.  Each record contains:

- ``name``         – unique identifier (slug-like, no spaces)
- ``display_name`` – human-readable label shown in the UI
- ``source``       – where the template came from (``"local"``, ``"comfyui"``, …)
- ``description``  – optional free-text description
- ``approved``     – bool; set by admin after reviewing/testing
- ``enabled``      – bool; admin can temporarily disable without removing approval
- ``filename``     – optional path to the JSON workflow file (relative to data/templates/)
- ``last_seen``    – ISO timestamp of last discovery/import
- ``created_at``   – ISO timestamp of first registration
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
TEMPLATES_FILE = DATA_DIR / "templates.json"
TEMPLATES_DIR = DATA_DIR / "templates"


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


def load_templates() -> list[dict[str, Any]]:
    if not TEMPLATES_FILE.exists():
        return []
    try:
        data = json.loads(TEMPLATES_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("template_registry: could not load templates: %s", exc)
    return []


def save_templates(templates: list[dict[str, Any]]) -> None:
    _ensure_data_dir()
    TEMPLATES_FILE.write_text(
        json.dumps(templates, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_approved_templates() -> list[dict[str, Any]]:
    """Return templates that are both approved and enabled (shown to normal users)."""
    return [
        t for t in load_templates() if t.get("approved") and t.get("enabled", True)
    ]


def get_all_templates() -> list[dict[str, Any]]:
    return load_templates()


def get_template(name: str) -> dict[str, Any] | None:
    for t in load_templates():
        if t.get("name") == name:
            return t
    return None


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

def register_template(
    name: str,
    display_name: str,
    source: str = "local",
    description: str = "",
    filename: str | None = None,
    approved: bool = False,
    enabled: bool = True,
) -> dict[str, Any]:
    """Register a new template or update last_seen if it already exists."""
    templates = load_templates()
    now = datetime.datetime.utcnow().isoformat()
    for t in templates:
        if t["name"] == name:
            t["last_seen"] = now
            t["display_name"] = display_name
            if filename is not None:
                t["filename"] = filename
            save_templates(templates)
            return t

    record: dict[str, Any] = {
        "name": name,
        "display_name": display_name,
        "source": source,
        "description": description,
        "filename": filename,
        "approved": approved,
        "enabled": enabled,
        "last_seen": now,
        "created_at": now,
    }
    templates.append(record)
    save_templates(templates)
    return record


def update_template(name: str, **fields: Any) -> dict[str, Any] | None:
    """Update allowed fields for a template record."""
    allowed = {"approved", "enabled", "display_name", "description"}
    templates = load_templates()
    for t in templates:
        if t["name"] == name:
            for key, value in fields.items():
                if key in allowed:
                    t[key] = value
            save_templates(templates)
            return t
    return None


def delete_template(name: str) -> bool:
    templates = load_templates()
    new = [t for t in templates if t["name"] != name]
    if len(new) == len(templates):
        return False
    save_templates(new)
    return True


# ---------------------------------------------------------------------------
# ComfyUI template discovery
# ---------------------------------------------------------------------------

async def discover_comfyui_templates(
    comfyui_base_url: str,
) -> tuple[list[dict[str, Any]], str | None]:
    """Try to fetch workflow templates from ComfyUI's built-in template API.

    Returns ``(discovered, error_message)``.  *discovered* is a list of
    newly-discovered template records (not yet saved); callers should persist
    them with :func:`register_template` as needed.  *error_message* is
    ``None`` on success or a human-readable string when no templates could be
    fetched.
    """
    import httpx  # local import to keep module import-time clean

    # ComfyUI has served workflow templates under different paths across versions.
    # Try them in order and return as soon as one succeeds.
    candidate_urls = [
        f"{comfyui_base_url}/api/workflow_templates",
        f"{comfyui_base_url}/workflow_templates",
    ]

    last_error: str | None = None
    for url in candidate_urls:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            # ComfyUI returns a list or dict of template metadata
            items: list[Any] = data if isinstance(data, list) else list(data.values())
            discovered: list[dict[str, Any]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                tname = item.get("name") or item.get("title") or ""
                if not tname:
                    continue
                slug = tname.lower().replace(" ", "_").replace("/", "_")
                discovered.append(
                    {
                        "name": slug,
                        "display_name": tname,
                        "source": "comfyui",
                        "description": item.get("description", ""),
                        "filename": None,
                    }
                )
            logger.info(
                "template_registry: discovered %d templates from ComfyUI (%s)",
                len(discovered),
                url,
            )
            return discovered, None
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "template_registry: ComfyUI template discovery failed (%s): %s",
                url,
                exc,
            )

    error_msg = (
        f"Kein kompatibles Template-Endpunkt gefunden. "
        f"Getestete URLs: {', '.join(candidate_urls)}. "
        f"Letzter Fehler: {last_error}"
    )
    return [], error_msg
