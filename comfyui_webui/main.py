from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

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


class GenerateRequest(BaseModel):
    prompt_de: str = Field(min_length=1)
    negative_prompt: str = ""
    ollama_model: str = Field(min_length=1)
    translated_prompt: str | None = None
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
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama nicht erreichbar: {exc}") from exc

    models = [item.get("name", "") for item in response.json().get("models", [])]
    models = [name for name in models if name]
    return {"models": models}


@app.post("/api/translate")
async def translate_prompt(payload: TranslateRequest) -> dict[str, str]:
    translated = await _translate_german_to_english(payload.prompt_de, payload.model)
    return {"translated_prompt": translated}


@app.get("/api/comfy/checkpoints")
async def get_comfy_checkpoints() -> dict[str, Any]:
    sources: list[str] = []
    checkpoints: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{COMFYUI_BASE_URL}/object_info/CheckpointLoaderSimple")
        response.raise_for_status()
        ckpt_names = (
            response.json()
            .get("CheckpointLoaderSimple", {})
            .get("input", {})
            .get("required", {})
            .get("ckpt_name", [[]])[0]
        )
        if isinstance(ckpt_names, list):
            checkpoints = [str(item) for item in ckpt_names if item]
            sources.append("/object_info/CheckpointLoaderSimple")
    except httpx.HTTPError:
        pass

    if not checkpoints:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{COMFYUI_BASE_URL}/models")
            response.raise_for_status()
            ckpt_names = response.json().get("checkpoints", [])
            if isinstance(ckpt_names, list):
                checkpoints = [str(item) for item in ckpt_names if item]
                sources.append("/models")
        except httpx.HTTPError:
            pass

    if not checkpoints:
        return {
            "checkpoints": [],
            "note": "Keine Checkpoints automatisch abrufbar. Bitte Namen manuell eingeben.",
            "sources": sources,
        }

    return {"checkpoints": checkpoints, "sources": sources}


@app.post("/api/generate")
async def generate_images(payload: GenerateRequest) -> dict[str, Any]:
    translated_prompt = payload.translated_prompt
    if not translated_prompt:
        translated_prompt = await _translate_german_to_english(payload.prompt_de, payload.ollama_model)

    workflow = _build_workflow(payload, translated_prompt)
    client_id = f"comfyui-webui-{uuid.uuid4()}"

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{COMFYUI_BASE_URL}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            )
        response.raise_for_status()
        prompt_id = response.json().get("prompt_id")
        if not prompt_id:
            raise HTTPException(status_code=502, detail="ComfyUI hat keine prompt_id zurückgegeben.")

        history_data = await _wait_for_history(prompt_id)
        images = _extract_images(history_data)
    except HTTPException:
        raise
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"ComfyUI-Fehler: {exc}") from exc

    if not images:
        raise HTTPException(status_code=502, detail="ComfyUI lieferte keine Bilder im History-Output.")

    image_urls = [
        f"/api/comfy/image?{urlencode({'filename': img['filename'], 'subfolder': img.get('subfolder', ''), 'type': img.get('type', 'output')})}"
        for img in images
    ]

    return {
        "translated_prompt": translated_prompt,
        "prompt_id": prompt_id,
        "images": image_urls,
    }


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


async def _translate_german_to_english(prompt_de: str, model: str) -> str:
    """Translate German prompt to English via Ollama.

    Tries /api/chat first (modern Ollama ≥ 0.1.14), then falls back to
    /api/generate for older installations.
    """
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


def _build_workflow(payload: GenerateRequest, translated_prompt: str) -> dict[str, dict[str, Any]]:
    workflow_template = _load_default_workflow()
    workflow = {key: {"class_type": node["class_type"], "inputs": dict(node["inputs"])} for key, node in workflow_template.items()}

    workflow["2"]["inputs"]["text"] = translated_prompt
    workflow["3"]["inputs"]["text"] = payload.negative_prompt

    if payload.checkpoint:
        workflow["1"]["inputs"]["ckpt_name"] = payload.checkpoint

    workflow["4"]["inputs"]["width"] = max(64, payload.width // 8 * 8)
    workflow["4"]["inputs"]["height"] = max(64, payload.height // 8 * 8)
    workflow["4"]["inputs"]["batch_size"] = max(1, min(payload.image_count, 8))

    seed = payload.seed if payload.seed >= 0 else int.from_bytes(os.urandom(8), "big")
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


async def _wait_for_history(prompt_id: str, timeout_seconds: int = 180) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        for _ in range(timeout_seconds):
            response = await client.get(f"{COMFYUI_BASE_URL}/history/{prompt_id}")
            response.raise_for_status()
            data = response.json()
            if data.get(prompt_id):
                return data[prompt_id]
            await asyncio.sleep(1)

    raise asyncio.TimeoutError(f"Timeout beim Warten auf ComfyUI-Ergebnis für {prompt_id}")


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
