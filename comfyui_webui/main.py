from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator
from urllib.parse import urlencode

import httpx
import websockets
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
async def get_ollama_models() -> dict[str, list[str]]:
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
async def translate_prompt(payload: TranslateRequest) -> dict[str, str]:
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
async def get_comfy_checkpoints() -> dict[str, Any]:
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
async def get_comfy_samplers() -> dict[str, list[str]]:
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
async def generate_images(payload: GenerateRequest) -> dict[str, Any]:
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
async def comfy_progress_sse(prompt_id: str, client_id: str, request: Request) -> StreamingResponse:
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
async def comfy_image_proxy(filename: str, subfolder: str = "", type: str = "output") -> Response:
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


def _build_workflow(payload: GenerateRequest, translated_prompt: str, translated_negative: str) -> dict[str, dict[str, Any]]:
    workflow_template = _load_default_workflow()
    workflow = {key: {"class_type": node["class_type"], "inputs": dict(node["inputs"])} for key, node in workflow_template.items()}

    workflow["2"]["inputs"]["text"] = translated_prompt
    workflow["3"]["inputs"]["text"] = translated_negative

    if payload.checkpoint:
        # Strip the [unet] tag added by get_comfy_checkpoints for display purposes
        is_unet = payload.checkpoint.startswith("[unet] ")
        model_name = payload.checkpoint.removeprefix("[unet] ")
        node1_type = workflow.get("1", {}).get("class_type", "")
        if node1_type in ("UNETLoader", "DiffusionModelLoader"):
            workflow["1"]["inputs"]["unet_name"] = model_name
        elif is_unet:
            # User selected a UNet/Diffusion model (e.g. FLUX, Zimage) but the
            # workflow template still uses CheckpointLoaderSimple, which cannot
            # load UNet-only models and causes ComfyUI to return 400 "Prompt
            # outputs failed validation".  Raise a clear error instead.
            raise ValueError(
                f"Das Modell '{model_name}' ist ein UNet-/Diffusion-Modell (z. B. FLUX, Zimage) "
                "und lässt sich nicht mit dem Standard-Template laden. "
                "Erstelle eine workflow_template.json mit einem UNETLoader- (oder "
                "DiffusionModelLoader-) Knoten als Node '1' sowie passenden "
                "CLIPLoader/DualCLIPLoader- und VAELoader-Knoten."
            )
        else:
            workflow["1"]["inputs"]["ckpt_name"] = model_name

    workflow["4"]["inputs"]["width"] = max(64, payload.width // 8 * 8)
    workflow["4"]["inputs"]["height"] = max(64, payload.height // 8 * 8)
    workflow["4"]["inputs"]["batch_size"] = max(1, min(payload.image_count, 8))

    seed = payload.seed if payload.seed >= 0 else int.from_bytes(os.urandom(4), "big")
    workflow["5"]["inputs"].update(
        {
            "seed": seed,
            "steps": max(1, min(payload.steps, 200)),
            "cfg": max(0.0, min(payload.cfg, 30.0)),
            "sampler_name": payload.sampler,
            "scheduler": payload.scheduler,
        }
    )

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
