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
- ``analysis``     – dict; result of deep workflow analysis (see :mod:`workflow_analyzer`)
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

import workflow_analyzer as _analyzer

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

def detect_model_type(workflow_json: dict) -> str:
    """Inspect a ComfyUI workflow JSON and return the required model type.

    Returns ``'unet'`` if a UNETLoader/DiffusionModelLoader node is found,
    ``'checkpoint'`` if a CheckpointLoaderSimple node is found, or ``'any'``
    otherwise.

    .. note::
        For full graph analysis use :func:`analyze_template_file` instead.
    """
    roles = _analyzer.analyze_workflow(workflow_json)
    return roles.model_type


def analyze_template_file(path: Path) -> dict[str, Any]:
    """Parse and deeply analyze a workflow JSON file.

    Returns a dict with keys:

    ``valid``
        ``True`` if the file parsed correctly as a JSON object.
    ``parse_error``
        Human-readable parse error string, or ``None``.
    ``analysis``
        :meth:`~workflow_analyzer.WorkflowRoles.to_dict` result, or ``None``
        when the file could not be parsed.
    ``model_type``
        ``"checkpoint"`` / ``"unet"`` / ``"any"`` derived from analysis, or
        ``"any"`` on parse failure.
    ``analyzed_at``
        ISO timestamp of analysis.
    """
    now = datetime.datetime.utcnow().isoformat()
    result: dict[str, Any] = {
        "valid": False,
        "parse_error": None,
        "analysis": None,
        "model_type": "any",
        "analyzed_at": now,
    }
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        # Sanitize exception message: strip file paths / internal details from OSError
        if isinstance(exc, OSError):
            safe_msg = f"Dateifehler: {exc.strerror or 'Unbekannter Fehler'}"
        elif isinstance(exc, json.JSONDecodeError):
            safe_msg = f"JSON-Fehler: {exc.msg} (Zeile {exc.lineno})"
        else:
            safe_msg = type(exc).__name__
        result["parse_error"] = safe_msg
        logger.warning("analyze_template_file: cannot parse %s: %s", path.name, exc)
        return result

    if not isinstance(data, dict):
        result["parse_error"] = "JSON ist kein Objekt (erwartet wird ein ComfyUI-Workflow-JSON)."
        return result

    result["valid"] = True
    roles = _analyzer.analyze_workflow(data)
    result["analysis"] = roles.to_dict()
    result["model_type"] = roles.model_type

    if not roles.is_usable:
        logger.warning(
            "analyze_template_file: %s – not usable: %s",
            path.name,
            "; ".join(roles.errors),
        )
    elif roles.warnings:
        logger.info(
            "analyze_template_file: %s – usable with %d warning(s): %s",
            path.name,
            len(roles.warnings),
            "; ".join(roles.warnings),
        )
    else:
        logger.debug("analyze_template_file: %s – OK", path.name)

    return result


def register_template(
    name: str,
    display_name: str,
    source: str = "local",
    description: str = "",
    filename: str | None = None,
    approved: bool = False,
    enabled: bool = True,
    model_type: str = "any",
    analysis: dict[str, Any] | None = None,
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
            # Update model_type if a specific type was detected, or if not yet set
            if model_type != "any" or "model_type" not in t:
                t["model_type"] = model_type
            if analysis is not None:
                t["analysis"] = analysis
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
        "model_type": model_type,
        "analysis": analysis,
        "last_seen": now,
        "created_at": now,
    }
    templates.append(record)
    save_templates(templates)
    return record


def update_template(name: str, **fields: Any) -> dict[str, Any] | None:
    """Update allowed fields for a template record."""
    allowed = {"approved", "enabled", "display_name", "description", "model_type", "analysis"}
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
# Local template file discovery
# ---------------------------------------------------------------------------

def discover_local_templates() -> list[dict[str, Any]]:
    """Scan ``data/templates/`` for ``.json`` workflow files and auto-register new ones.

    Files found here are registered as *approved* + *enabled* because the admin
    intentionally placed them in that directory.  If a template with the same
    slug already exists its ``filename`` reference is updated but its
    ``approved``/``enabled`` flags are **not** changed (so a deliberately
    disabled template stays disabled).

    Each file is deeply validated via :func:`analyze_template_file` and the
    result is stored in the template record's ``analysis`` field.
    """
    _ensure_data_dir()
    registered: list[dict[str, Any]] = []
    for json_file in sorted(TEMPLATES_DIR.glob("*.json")):
        stem = json_file.stem
        slug = stem.lower().replace(" ", "_").replace("-", "_")
        display_name = stem.replace("_", " ").replace("-", " ").title()

        validation = analyze_template_file(json_file)
        model_type = validation.get("model_type", "any")
        analysis_meta = {
            "is_usable": validation["analysis"]["is_usable"] if validation.get("analysis") else False,
            "model_type": model_type,
            "sampler_count": validation["analysis"].get("sampler_count", 0) if validation.get("analysis") else 0,
            "model_loader_count": validation["analysis"].get("model_loader_count", 0) if validation.get("analysis") else 0,
            "positive_clip_count": validation["analysis"].get("positive_clip_count", 0) if validation.get("analysis") else 0,
            "negative_is_zero_out": validation["analysis"].get("negative_is_zero_out", False) if validation.get("analysis") else False,
            "is_potentially_img2img": validation["analysis"].get("is_potentially_img2img", False) if validation.get("analysis") else False,
            "warnings": validation["analysis"].get("warnings", []) if validation.get("analysis") else [],
            "errors": validation["analysis"].get("errors", [validation.get("parse_error", "Datei konnte nicht gelesen werden")]) if not validation.get("valid") else (validation["analysis"].get("errors", []) if validation.get("analysis") else []),
            "analyzed_at": validation.get("analyzed_at"),
            "parse_error": validation.get("parse_error"),
        }

        if not validation.get("valid"):
            logger.warning(
                "discover_local_templates: %s konnte nicht geparst werden: %s",
                json_file.name,
                validation.get("parse_error"),
            )
            # Still register it so admins can see it and the parse error
            existing = get_template(slug)
            if existing is None:
                record = register_template(
                    name=slug,
                    display_name=display_name,
                    source="local",
                    description=f"Lokales Workflow-Template: {json_file.name}",
                    filename=json_file.name,
                    approved=False,  # don't auto-approve broken templates
                    enabled=False,
                    model_type="any",
                    analysis=analysis_meta,
                )
                logger.info("discover_local_templates: registriert (fehlerhaft) %r aus %s", slug, json_file.name)
            else:
                record = register_template(
                    name=slug,
                    display_name=existing.get("display_name", display_name),
                    source=existing.get("source", "local"),
                    filename=json_file.name,
                    model_type="any",
                    analysis=analysis_meta,
                )
            registered.append(record)
            continue

        existing = get_template(slug)
        if existing is None:
            record = register_template(
                name=slug,
                display_name=display_name,
                source="local",
                description=f"Lokales Workflow-Template: {json_file.name}",
                filename=json_file.name,
                approved=True,
                enabled=True,
                model_type=model_type,
                analysis=analysis_meta,
            )
            logger.info("discover_local_templates: registriert %r aus %s (model_type=%s)", slug, json_file.name, model_type)
        else:
            record = register_template(
                name=slug,
                display_name=existing.get("display_name", display_name),
                source=existing.get("source", "local"),
                filename=json_file.name,
                model_type=model_type,
                analysis=analysis_meta,
            )
            logger.debug("discover_local_templates: aktualisiert %r", slug)
        registered.append(record)
    return registered


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
