from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator
from urllib.parse import urlencode

import httpx
import websockets
from fastapi import Cookie, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import auth as _auth
import template_registry as _registry

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
COMFYUI_BASE_URL = os.getenv("COMFYUI_BASE_URL", "http://127.0.0.1:8188").rstrip("/")

DEFAULT_WORKFLOW = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ""}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
    "4": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 30,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
        },
    },
    "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
    "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ollama_comfyui", "images": ["6", 0]}},
}
WORKFLOW_TEMPLATE_PATH = APP_DIR / "workflow_template.json"


class TranslateRequest(BaseModel):
    prompt_de: str = Field(min_length=1)
    model: str = Field(min_length=1)
    context_prompt: str | None = None


class GenerateRequest(BaseModel):
    prompt_de: str = Field(min_length=1)
    negative_prompt: str = ""
    ollama_model: str = Field(min_length=1)
    translated_prompt: str | None = None
    translated_negative_prompt: str | None = None
    context_prompt: str | None = None
    checkpoint: str | None = None
    workflow_template: str | None = None  # name of template from registry
    steps: int = 30
    cfg: float = 7.0
    seed: int = -1
    width: int = 1024
    height: int = 1024
    sampler: str = "euler"
    scheduler: str = "normal"
    image_count: int = 1


app = FastAPI(title="ComfyUI Ollama WebUI", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# In-memory session store  {token: {"username": str, "role": str}}
# Sessions are intentionally not persisted – users must log in again after
# a server restart.  This is fine for a local self-hosted app.
# ---------------------------------------------------------------------------
_SESSION_COOKIE = "ki_session"
_sessions: dict[str, dict[str, str]] = {}


@app.on_event("startup")
async def _startup() -> None:
    """Bootstrap admin account and seed example templates on first run."""
    bootstrap_credential = _auth.bootstrap_admin()
    if bootstrap_credential:
        # Print the one-time bootstrap credential to stdout only.
        # We deliberately avoid logger.* here so it is not captured in log files.
        _lines = [
            "=" * 60,
            "  FIRST START – admin account created",
            "  username   : admin",
            "  credential : " + bootstrap_credential,
            "  See comfyui_webui/data/bootstrap_credentials.txt",
            "  Delete that file after first login!",
            "=" * 60,
        ]
        print("\n" + "\n".join(_lines) + "\n", flush=True)
    # Always ensure the built-in "default" template exists
    if _registry.get_template("default") is None:
        _registry.register_template(
            name="default",
            display_name="Standard (CheckpointLoaderSimple)",
            source="local",
            description="Built-in default workflow (CheckpointLoaderSimple + KSampler).",
            filename=None,
            approved=True,
            enabled=True,
        )
    # Auto-discover workflow JSON files placed in data/templates/
    local_found = _registry.discover_local_templates()
    if local_found:
        logger.info("startup: discovered %d local template(s)", len(local_found))


# ---------------------------------------------------------------------------
# Auth helpers / FastAPI dependencies
# ---------------------------------------------------------------------------

def _get_session(ki_session: str | None = Cookie(default=None)) -> dict[str, str] | None:
    if ki_session and ki_session in _sessions:
        return _sessions[ki_session]
    return None


def require_user(session: dict[str, str] | None = Depends(_get_session)) -> dict[str, str]:
    if session is None:
        raise HTTPException(status_code=401, detail="Nicht eingeloggt.")
    return session


def require_admin(session: dict[str, str] = Depends(require_user)) -> dict[str, str]:
    if session.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin-Berechtigung erforderlich.")
    return session


# ---------------------------------------------------------------------------
# Auth request/response models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=8)
    role: str = "user"


class UpdateUserRequest(BaseModel):
    disabled: bool | None = None
    role: str | None = None


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1)
    new_password: str = Field(min_length=8)


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/api/auth/login")
async def login(payload: LoginRequest, response: Response) -> dict[str, Any]:
    user = _auth.authenticate(payload.username, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Ungültige Anmeldedaten.")
    token = secrets.token_urlsafe(32)
    _sessions[token] = {"username": user["username"], "role": user["role"]}
    response.set_cookie(
        key=_SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=86400,  # 24 h
    )
    return {"username": user["username"], "role": user["role"]}


@app.post("/api/auth/logout")
async def logout(
    response: Response,
    ki_session: str | None = Cookie(default=None),
) -> dict[str, str]:
    if ki_session and ki_session in _sessions:
        del _sessions[ki_session]
    response.delete_cookie(_SESSION_COOKIE)
    return {"status": "ok"}


@app.get("/api/auth/me")
async def me(session: dict[str, str] | None = Depends(_get_session)) -> dict[str, Any]:
    if session is None:
        raise HTTPException(status_code=401, detail="Nicht eingeloggt.")
    return {"username": session["username"], "role": session["role"]}


@app.post("/api/auth/change_password")
async def change_password(
    payload: ChangePasswordRequest,
    session: dict[str, str] = Depends(require_user),
) -> dict[str, str]:
    username = session["username"]
    # Verify the current password before allowing the change
    if _auth.authenticate(username, payload.current_password) is None:
        raise HTTPException(status_code=400, detail="Aktuelles Passwort ist falsch.")
    if not _auth.change_password(username, payload.new_password):
        raise HTTPException(status_code=404, detail="Benutzer nicht gefunden.")
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Admin – user management
# ---------------------------------------------------------------------------

@app.get("/api/admin/users")
async def admin_list_users(
    _: dict[str, str] = Depends(require_admin),
) -> dict[str, Any]:
    users = [_auth.public_user(u) for u in _auth.load_users()]
    return {"users": users}


@app.post("/api/admin/users", status_code=201)
async def admin_create_user(
    payload: CreateUserRequest,
    _: dict[str, str] = Depends(require_admin),
) -> dict[str, Any]:
    if payload.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="Ungültige Rolle. Erlaubt: admin, user")
    try:
        user = _auth.create_user(payload.username, payload.password, payload.role)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _auth.public_user(user)


@app.patch("/api/admin/users/{username}")
async def admin_update_user(
    username: str,
    payload: UpdateUserRequest,
    current: dict[str, str] = Depends(require_admin),
) -> dict[str, Any]:
    # Prevent admin from disabling themselves
    if username == current["username"] and payload.disabled is True:
        raise HTTPException(status_code=400, detail="Du kannst dich nicht selbst deaktivieren.")
    fields: dict[str, Any] = {}
    if payload.disabled is not None:
        fields["disabled"] = payload.disabled
    if payload.role is not None:
        if payload.role not in ("admin", "user"):
            raise HTTPException(status_code=400, detail="Ungültige Rolle.")
        fields["role"] = payload.role
    updated = _auth.update_user(username, **fields)
    if updated is None:
        raise HTTPException(status_code=404, detail="Benutzer nicht gefunden.")
    return _auth.public_user(updated)


# ---------------------------------------------------------------------------
# Template registry endpoints
# ---------------------------------------------------------------------------

class RegisterTemplateRequest(BaseModel):
    name: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    source: str = "local"
    description: str = ""
    approved: bool = False
    enabled: bool = True


class UpdateTemplateRequest(BaseModel):
    approved: bool | None = None
    enabled: bool | None = None
    display_name: str | None = None
    description: str | None = None


@app.get("/api/templates")
async def list_templates_for_user(
    _: dict[str, str] = Depends(require_user),
) -> dict[str, Any]:
    """Return approved + enabled templates (visible to all authenticated users)."""
    templates = _registry.get_approved_templates()
    return {"templates": templates}


@app.get("/api/admin/templates")
async def admin_list_templates(
    _: dict[str, str] = Depends(require_admin),
) -> dict[str, Any]:
    """Return all templates including unapproved ones (admin only)."""
    templates = _registry.get_all_templates()
    return {"templates": templates}


@app.post("/api/admin/templates", status_code=201)
async def admin_register_template(
    payload: RegisterTemplateRequest,
    _: dict[str, str] = Depends(require_admin),
) -> dict[str, Any]:
    record = _registry.register_template(
        name=payload.name,
        display_name=payload.display_name,
        source=payload.source,
        description=payload.description,
        approved=payload.approved,
        enabled=payload.enabled,
    )
    return record


@app.patch("/api/admin/templates/{name}")
async def admin_update_template(
    name: str,
    payload: UpdateTemplateRequest,
    _: dict[str, str] = Depends(require_admin),
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        k: v for k, v in payload.model_dump().items() if v is not None
    }
    updated = _registry.update_template(name, **fields)
    if updated is None:
        raise HTTPException(status_code=404, detail="Template nicht gefunden.")
    return updated


@app.delete("/api/admin/templates/{name}")
async def admin_delete_template(
    name: str,
    _: dict[str, str] = Depends(require_admin),
) -> dict[str, str]:
    if not _registry.delete_template(name):
        raise HTTPException(status_code=404, detail="Template nicht gefunden.")
    return {"status": "deleted"}


@app.post("/api/admin/templates/discover")
async def admin_discover_templates(
    _: dict[str, str] = Depends(require_admin),
) -> dict[str, Any]:
    """Attempt to fetch workflow templates from ComfyUI and add unseen ones."""
    discovered, error_msg = await _registry.discover_comfyui_templates(COMFYUI_BASE_URL)
    added = 0
    for item in discovered:
        existing = _registry.get_template(item["name"])
        if existing is None:
            _registry.register_template(
                name=item["name"],
                display_name=item["display_name"],
                source=item["source"],
                description=item.get("description", ""),
            )
            added += 1
    result: dict[str, Any] = {"discovered": len(discovered), "added": added}
    if error_msg:
        result["error"] = error_msg
    return result


@app.post("/api/admin/templates/discover_local")
async def admin_discover_local_templates(
    _: dict[str, str] = Depends(require_admin),
) -> dict[str, Any]:
    """Scan ``data/templates/`` for workflow JSON files and auto-register them.

    Drop any ``.json`` ComfyUI workflow file into the ``comfyui_webui/data/templates/``
    directory and click this button – the file will be registered as an approved
    template immediately.
    """
    registered = _registry.discover_local_templates()
    return {"found": len(registered), "templates": [t["name"] for t in registered]}


@app.post("/api/admin/templates/upload", status_code=201)
async def admin_upload_template(
    file: UploadFile = File(...),
    _: dict[str, str] = Depends(require_admin),
) -> dict[str, Any]:
    """Upload a ComfyUI workflow JSON file and register it as a template.

    The file is saved to ``data/templates/`` and immediately registered as an
    approved + enabled template (same behaviour as *Lokale Templates laden*).
    """
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Nur JSON-Dateien erlaubt.")

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:  # 10 MB safety limit
        raise HTTPException(status_code=413, detail="Datei zu groß (max. 10 MB).")

    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Ungültige JSON-Datei: {exc}") from exc

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="JSON muss ein Objekt sein (ComfyUI workflow).")

    # Sanitise filename: keep only safe chars
    safe_name = Path(file.filename).name
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in ("_", "-", "."))
    if not safe_name:
        safe_name = "workflow.json"
    if not safe_name.lower().endswith(".json"):
        safe_name += ".json"

    dest = _registry.TEMPLATES_DIR / safe_name
    _registry.TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)

    # Register (same logic as discover_local_templates for a single file)
    stem = Path(safe_name).stem
    slug = stem.lower().replace(" ", "_").replace("-", "_")
    display_name = stem.replace("_", " ").replace("-", " ").title()
    existing = _registry.get_template(slug)
    if existing is None:
        record = _registry.register_template(
            name=slug,
            display_name=display_name,
            source="local",
            description=f"Hochgeladen: {safe_name}",
            filename=safe_name,
            approved=True,
            enabled=True,
        )
    else:
        record = _registry.register_template(
            name=slug,
            display_name=existing.get("display_name", display_name),
            source=existing.get("source", "local"),
            filename=safe_name,
        )
    return record


def _load_default_workflow() -> dict[str, dict[str, Any]]:
    if not WORKFLOW_TEMPLATE_PATH.exists():
        return DEFAULT_WORKFLOW
    try:
        data = json.loads(WORKFLOW_TEMPLATE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data  # type: ignore[return-value]
    except (OSError, json.JSONDecodeError):
        pass
    return DEFAULT_WORKFLOW


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def get_config() -> dict[str, str]:
    return {
        "ollama_base_url": OLLAMA_BASE_URL,
        "comfyui_base_url": COMFYUI_BASE_URL,
    }


@app.get("/api/ollama/models")
async def get_ollama_models(
    _: dict[str, str] = Depends(require_user),
) -> dict[str, list[str]]:
    logger.info("get_ollama_models: querying %s/api/tags", OLLAMA_BASE_URL)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
        models = [item.get("name", "") for item in response.json().get("models", [])]
        models = [name for name in models if name]
        logger.info("get_ollama_models: found %d models", len(models))
    except httpx.HTTPError as exc:
        logger.warning("get_ollama_models: HTTP error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Ollama nicht erreichbar: {exc}") from exc
    except Exception as exc:
        logger.warning("get_ollama_models: unexpected error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Ollama-Antwort konnte nicht gelesen werden: {exc}") from exc

    return {"models": models}


@app.post("/api/translate")
async def translate_prompt(
    payload: TranslateRequest,
    _: dict[str, str] = Depends(require_user),
) -> dict[str, str]:
    translated = await _translate_german_to_english(payload.prompt_de, payload.model, payload.context_prompt)
    return {"translated_prompt": translated}


def _extract_object_info_names(data: Any, node_key: str, input_key: str) -> list[str]:
    """Extract the name list from a ComfyUI /object_info/<NodeType> response."""
    raw = (
        data.get(node_key, {})
        .get("input", {})
        .get("required", {})
        .get(input_key, [[]])[0]
    )
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    return []


@app.get("/api/comfy/checkpoints")
async def get_comfy_checkpoints(
    _: dict[str, str] = Depends(require_user),
) -> dict[str, Any]:
    sources: list[str] = []
    checkpoints: list[str] = []
    unet_models: list[str] = []

    # ── Checkpoints (CheckpointLoaderSimple) ──────────────────────────────
    logger.info("get_comfy_checkpoints: trying %s/object_info/CheckpointLoaderSimple", COMFYUI_BASE_URL)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{COMFYUI_BASE_URL}/object_info/CheckpointLoaderSimple")
        response.raise_for_status()
        names = _extract_object_info_names(response.json(), "CheckpointLoaderSimple", "ckpt_name")
        if names:
            checkpoints = names
            sources.append("/object_info/CheckpointLoaderSimple")
            logger.info("get_comfy_checkpoints: found %d checkpoints via object_info", len(checkpoints))
    except Exception as exc:
        logger.warning("get_comfy_checkpoints: object_info failed: %s", exc)

    # Newer ComfyUI API – checkpoints
    if not checkpoints:
        logger.info("get_comfy_checkpoints: trying %s/api/models/checkpoints", COMFYUI_BASE_URL)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{COMFYUI_BASE_URL}/api/models/checkpoints")
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                checkpoints = [str(item) for item in data if item]
                sources.append("/api/models/checkpoints")
                logger.info("get_comfy_checkpoints: found %d checkpoints via /api/models/checkpoints", len(checkpoints))
        except Exception as exc:
            logger.warning("get_comfy_checkpoints: /api/models/checkpoints failed: %s", exc)

    if not checkpoints:
        logger.info("get_comfy_checkpoints: trying %s/models", COMFYUI_BASE_URL)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{COMFYUI_BASE_URL}/models")
            response.raise_for_status()
            ckpt_names = response.json().get("checkpoints", [])
            if isinstance(ckpt_names, list):
                checkpoints = [str(item) for item in ckpt_names if item]
                sources.append("/models")
                logger.info("get_comfy_checkpoints: found %d checkpoints via /models", len(checkpoints))
        except Exception as exc:
            logger.warning("get_comfy_checkpoints: /models failed: %s", exc)

    # ── UNet / Diffusion models (UNETLoader / DiffusionModelLoader) ───────
    logger.info("get_comfy_checkpoints: trying %s/object_info/UNETLoader", COMFYUI_BASE_URL)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{COMFYUI_BASE_URL}/object_info/UNETLoader")
        response.raise_for_status()
        names = _extract_object_info_names(response.json(), "UNETLoader", "unet_name")
        if names:
            unet_models = names
            sources.append("/object_info/UNETLoader")
            logger.info("get_comfy_checkpoints: found %d unet models via UNETLoader object_info", len(unet_models))
    except Exception as exc:
        logger.warning("get_comfy_checkpoints: UNETLoader object_info failed: %s", exc)

    # Newer ComfyUI API – unet models
    if not unet_models:
        logger.info("get_comfy_checkpoints: trying %s/api/models/unet", COMFYUI_BASE_URL)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{COMFYUI_BASE_URL}/api/models/unet")
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                unet_models = [str(item) for item in data if item]
                sources.append("/api/models/unet")
                logger.info("get_comfy_checkpoints: found %d unet models via /api/models/unet", len(unet_models))
        except Exception as exc:
            logger.warning("get_comfy_checkpoints: /api/models/unet failed: %s", exc)

    # ── Merge: unet models are shown tagged so the UI can distinguish them ─
    unet_set = set(unet_models)
    all_models = checkpoints + [f"[unet] {name}" for name in unet_models if name not in set(checkpoints)]

    note = ""
    if not all_models:
        logger.warning("get_comfy_checkpoints: no models found via any source (ComfyUI URL: %s)", COMFYUI_BASE_URL)
        note = (
            "Keine Modelle gefunden. "
            "Checkpoints müssen in ComfyUI/models/checkpoints/ liegen, "
            "UNet-Modelle (FLUX/Zimage) in ComfyUI/models/unet/ – "
            "oder über extra_model_paths.yaml eingebunden sein."
        )
    elif unet_models:
        logger.info(
            "get_comfy_checkpoints: total %d checkpoints + %d unet models",
            len(checkpoints), len(unet_models),
        )

    return {
        "checkpoints": all_models,
        "unet_models": list(unet_set),
        "sources": sources,
        "note": note,
    }


_DEFAULT_SAMPLERS = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp",
    "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast",
    "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v", "deis",
    "ddim", "uni_pc", "uni_pc_bh2",
]
_DEFAULT_SCHEDULERS = [
    "normal", "karras", "exponential", "sgm_uniform", "simple",
    "ddim_uniform", "beta", "linear_quadratic", "kl_optimal",
]


@app.get("/api/comfy/samplers")
async def get_comfy_samplers(
    _: dict[str, str] = Depends(require_user),
) -> dict[str, list[str]]:
    samplers = list(_DEFAULT_SAMPLERS)
    schedulers = list(_DEFAULT_SCHEDULERS)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{COMFYUI_BASE_URL}/object_info/KSampler")
        response.raise_for_status()
        info = response.json().get("KSampler", {}).get("input", {}).get("required", {})
        s = info.get("sampler_name", [[]])[0]
        sc = info.get("scheduler", [[]])[0]
        if isinstance(s, list) and s:
            samplers = [str(x) for x in s if x]
        if isinstance(sc, list) and sc:
            schedulers = [str(x) for x in sc if x]
    except Exception:
        pass
    return {"samplers": samplers, "schedulers": schedulers}


@app.post("/api/generate")
async def generate_images(
    payload: GenerateRequest,
    _: dict[str, str] = Depends(require_user),
) -> dict[str, Any]:
    translated_prompt = payload.translated_prompt
    if not translated_prompt:
        translated_prompt = await _translate_german_to_english(
            payload.prompt_de, payload.ollama_model, payload.context_prompt
        )

    translated_negative = payload.translated_negative_prompt
    if payload.negative_prompt and not translated_negative:
        translated_negative = await _translate_german_to_english(
            payload.negative_prompt, payload.ollama_model
        )

    try:
        workflow = _build_workflow(payload, translated_prompt, translated_negative or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    client_id = f"comfyui-webui-{uuid.uuid4()}"

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{COMFYUI_BASE_URL}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            )
        if not response.is_success:
            # Capture ComfyUI's error body so the user sees the real validation message
            try:
                body = response.json()
                comfy_error = (
                    body.get("error", {}).get("message")
                    or body.get("error")
                    or body.get("detail")
                    or str(body)
                )
            except Exception:
                comfy_error = response.text[:500] or f"HTTP {response.status_code}"
            raise HTTPException(
                status_code=502,
                detail=f"ComfyUI-Fehler ({response.status_code}): {comfy_error}",
            )
        prompt_id = response.json().get("prompt_id")
        if not prompt_id:
            raise HTTPException(status_code=502, detail="ComfyUI hat keine prompt_id zurückgegeben.")
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"ComfyUI nicht erreichbar: {exc}") from exc

    return {
        "translated_prompt": translated_prompt,
        "translated_negative_prompt": translated_negative or "",
        "prompt_id": prompt_id,
        "client_id": client_id,
    }


@app.get("/api/comfy/progress/{prompt_id}")
async def comfy_progress_sse(
    prompt_id: str,
    client_id: str,
    request: Request,
    _: dict[str, str] = Depends(require_user),
) -> StreamingResponse:
    async def event_stream() -> AsyncGenerator[str, None]:
        # Initial queue-position check via REST
        try:
            async with httpx.AsyncClient(timeout=5.0) as http:
                q = await http.get(f"{COMFYUI_BASE_URL}/queue")
                pending = [item[1] for item in q.json().get("queue_pending", [])]
                pos: int | None = (pending.index(prompt_id) + 1) if prompt_id in pending else None
                yield f"data: {json.dumps({'type': 'queued', 'position': pos})}\n\n"
        except Exception:
            yield f"data: {json.dumps({'type': 'queued'})}\n\n"

        ws_url = (
            COMFYUI_BASE_URL.replace("http://", "ws://").replace("https://", "wss://")
            + f"/ws?clientId={client_id}"
        )
        start_time: float | None = None

        try:
            async with websockets.connect(ws_url, ping_interval=20) as ws:
                async for raw in ws:
                    if await request.is_disconnected():
                        return
                    if isinstance(raw, bytes):
                        continue  # skip binary preview frames
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    mtype = msg.get("type", "")
                    mdata = msg.get("data") or {}
                    mpid = mdata.get("prompt_id") if isinstance(mdata, dict) else None

                    # Filter messages for other prompts
                    if mpid and mpid != prompt_id:
                        continue

                    if mtype == "execution_start":
                        start_time = asyncio.get_event_loop().time()
                        yield f"data: {json.dumps({'type': 'start'})}\n\n"

                    elif mtype == "progress":
                        step = int(mdata.get("value", 0))
                        total = int(mdata.get("max", 1))
                        eta: int | None = None
                        if start_time and step > 0 and step < total:
                            elapsed = asyncio.get_event_loop().time() - start_time
                            eta = round(elapsed / step * (total - step))
                        yield f"data: {json.dumps({'type': 'progress', 'step': step, 'max': total, 'eta': eta})}\n\n"

                    elif mtype == "execution_success" or (
                        mtype == "executing" and isinstance(mdata, dict) and mdata.get("node") is None
                    ):
                        break

                    elif mtype == "execution_error":
                        err = mdata.get("exception_message", "Unbekannter Fehler") if isinstance(mdata, dict) else "Fehler"
                        yield f"data: {json.dumps({'type': 'error', 'message': str(err)})}\n\n"
                        return

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': f'WebSocket-Fehler: {exc}'})}\n\n"
            return

        # Fetch images from history
        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                for _ in range(20):
                    resp = await http.get(f"{COMFYUI_BASE_URL}/history/{prompt_id}")
                    resp.raise_for_status()
                    data = resp.json()
                    if data.get(prompt_id):
                        images = _extract_images(data[prompt_id])
                        image_urls = [
                            f"/api/comfy/image?{urlencode({'filename': img['filename'], 'subfolder': img.get('subfolder', ''), 'type': img.get('type', 'output')})}"
                            for img in images
                        ]
                        yield f"data: {json.dumps({'type': 'done', 'images': image_urls})}\n\n"
                        return
                    await asyncio.sleep(0.5)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Bilder nicht in History gefunden.'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/comfy/image")
async def comfy_image_proxy(
    filename: str,
    subfolder: str = "",
    type: str = "output",
    _: dict[str, str] = Depends(require_user),
) -> Response:
    params = {"filename": filename, "subfolder": subfolder, "type": type}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{COMFYUI_BASE_URL}/view", params=params)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Bild konnte nicht geladen werden: {exc}") from exc

    content_type = response.headers.get("content-type", "image/png")
    return Response(content=response.content, media_type=content_type)


async def _translate_german_to_english(prompt_de: str, model: str, context_prompt: str | None = None) -> str:
    """Translate German prompt to English via Ollama, optionally refining an existing prompt.

    Tries /api/chat first (modern Ollama ≥ 0.1.14), then falls back to
    /api/generate for older installations.
    """
    if context_prompt:
        instruction = (
            "You previously created an image with this English prompt:\n"
            f'"{context_prompt}"\n\n'
            "The user wants to modify it with this German instruction:\n"
            f'"{prompt_de}"\n\n'
            "Return only the updated English image prompt. No explanations, no quotes, no additional text."
        )
    else:
        instruction = (
            "Translate the following German image prompt into natural, precise English "
            "for text-to-image models. Return only the English prompt, no explanations, "
            "no quotes, no additional text.\n\n"
            f"German: {prompt_de}"
        )

    # --- Primary: /api/chat (supported by all current Ollama versions) ---
    chat_url = f"{OLLAMA_BASE_URL}/api/chat"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                chat_url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": instruction}],
                    "stream": False,
                    "keep_alive": -1,
                    "options": {"temperature": 0.1},
                },
            )
        if response.status_code < 400:
            translated = (
                response.json()
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if translated:
                return translated
    except httpx.HTTPError:
        pass  # fall through to /api/generate

    # --- Fallback: /api/generate (older Ollama / alternative endpoint) ---
    generate_url = f"{OLLAMA_BASE_URL}/api/generate"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                generate_url,
                json={
                    "model": model,
                    "prompt": instruction,
                    "stream": False,
                    "keep_alive": -1,
                    "options": {"temperature": 0.1},
                },
            )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Ollama nicht erreichbar oder Fehler beim Übersetzen. "
                f"Geprüfte URLs: {chat_url} und {generate_url}. "
                f"Fehler: {exc}"
            ),
        ) from exc

    translated = response.json().get("response", "").strip()
    if not translated:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama lieferte keinen Übersetzungstext (Modell: {model}, URL: {generate_url}).",
        )
    return translated


def _find_node_by_class(
    workflow: dict[str, Any], *class_types: str
) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    """Return the first ``(key, node)`` whose ``class_type`` matches any of *class_types*."""
    for key, node in workflow.items():
        if node.get("class_type") in class_types:
            return key, node
    return None, None


def _resolve_ref(
    workflow: dict[str, Any], ref: Any
) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    """Follow a ComfyUI node-reference ``[node_id, slot]`` and return ``(key, node)``."""
    if isinstance(ref, list) and len(ref) >= 1:
        key = str(ref[0])
        if key in workflow:
            return key, workflow[key]
    return None, None


def _build_workflow(payload: GenerateRequest, translated_prompt: str, translated_negative: str) -> dict[str, dict[str, Any]]:
    # Resolve workflow template: prefer registry-selected template file, then
    # fall back to the global workflow_template.json / built-in default.
    workflow_json: dict[str, Any] | None = None
    if payload.workflow_template and payload.workflow_template != "default":
        record = _registry.get_template(payload.workflow_template)
        if record and record.get("filename"):
            tpath = _registry.TEMPLATES_DIR / record["filename"]
            try:
                data = json.loads(tpath.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    workflow_json = data
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("_build_workflow: could not load template file %s: %s", tpath, exc)

    if workflow_json is None:
        workflow_json = _load_default_workflow()

    # Deep-copy: only keep class_type and inputs (drop _meta and other UI-only keys)
    workflow: dict[str, dict[str, Any]] = {
        key: {"class_type": node["class_type"], "inputs": dict(node["inputs"])}
        for key, node in workflow_json.items()
        if "class_type" in node and "inputs" in node
    }

    # ── Locate KSampler (or KSamplerAdvanced) ──────────────────────────────
    ksampler_key, ksampler_node = _find_node_by_class(workflow, "KSampler", "KSamplerAdvanced")
    if ksampler_node is None:
        raise ValueError(
            "Kein KSampler-Knoten im Workflow gefunden. "
            "Das Template muss einen KSampler- oder KSamplerAdvanced-Knoten enthalten."
        )

    # ── Inject positive prompt ──────────────────────────────────────────────
    pos_key, pos_node = _resolve_ref(workflow, ksampler_node["inputs"].get("positive"))
    if pos_node and pos_node.get("class_type") == "CLIPTextEncode":
        pos_node["inputs"]["text"] = translated_prompt
    else:
        # Fallback: first CLIPTextEncode in the workflow
        for node in workflow.values():
            if node.get("class_type") == "CLIPTextEncode":
                node["inputs"]["text"] = translated_prompt
                break

    # ── Inject negative prompt (only when target IS a CLIPTextEncode) ───────
    # Workflows that use ConditioningZeroOut or similar as "negative" are left
    # untouched – their negative conditioning is intentionally fixed.
    neg_key, neg_node = _resolve_ref(workflow, ksampler_node["inputs"].get("negative"))
    if neg_node and neg_node.get("class_type") == "CLIPTextEncode":
        neg_node["inputs"]["text"] = translated_negative

    # ── KSampler generation params ──────────────────────────────────────────
    seed = payload.seed if payload.seed >= 0 else int.from_bytes(os.urandom(4), "big")
    is_advanced = ksampler_node.get("class_type") == "KSamplerAdvanced"
    ksampler_node["inputs"].update(
        {
            ("noise_seed" if is_advanced else "seed"): seed,
            "steps": max(1, min(payload.steps, 200)),
            "cfg": max(0.0, min(payload.cfg, 30.0)),
            "sampler_name": payload.sampler,
            "scheduler": payload.scheduler,
        }
    )

    # ── Latent image dimensions ─────────────────────────────────────────────
    _LATENT_TYPES = ("EmptyLatentImage", "EmptySD3LatentImage", "EmptyHunyuanLatentVideo")
    latent_key, latent_node = _resolve_ref(workflow, ksampler_node["inputs"].get("latent_image"))
    if latent_node and latent_node.get("class_type") in _LATENT_TYPES:
        target = latent_node
    else:
        # Fallback: first EmptyLatentImage-like node in the workflow
        _, target = _find_node_by_class(workflow, *_LATENT_TYPES)
    if target is not None:
        target["inputs"]["width"] = max(64, payload.width // 8 * 8)
        target["inputs"]["height"] = max(64, payload.height // 8 * 8)
        target["inputs"]["batch_size"] = max(1, min(payload.image_count, 8))

    # ── Model loader ────────────────────────────────────────────────────────
    if payload.checkpoint:
        is_unet = payload.checkpoint.startswith("[unet] ")
        model_name = payload.checkpoint.removeprefix("[unet] ")

        unet_key, unet_node = _find_node_by_class(workflow, "UNETLoader", "DiffusionModelLoader")
        ckpt_key, ckpt_node = _find_node_by_class(workflow, "CheckpointLoaderSimple")

        if unet_node is not None:
            unet_node["inputs"]["unet_name"] = model_name
        elif ckpt_node is not None:
            if is_unet:
                raise ValueError(
                    f"Das Modell '{model_name}' ist ein UNet-/Diffusion-Modell (z. B. FLUX, Zimage) "
                    "und lässt sich nicht mit CheckpointLoaderSimple laden. "
                    "Bitte ein Template mit einem UNETLoader- oder DiffusionModelLoader-Knoten verwenden."
                )
            ckpt_node["inputs"]["ckpt_name"] = model_name

    return workflow


def _extract_images(history_data: dict[str, Any]) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    outputs = history_data.get("outputs", {})
    if not isinstance(outputs, dict):
        return images

    for node_output in outputs.values():
        node_images = node_output.get("images", []) if isinstance(node_output, dict) else []
        for image in node_images:
            filename = image.get("filename")
            if filename:
                images.append(
                    {
                        "filename": str(filename),
                        "subfolder": str(image.get("subfolder", "")),
                        "type": str(image.get("type", "output")),
                    }
                )
    return images
