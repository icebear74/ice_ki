"""HTTP API of the character extraction service.

Endpoints
---------
GET  /healthz   liveness/readiness probe
GET  /config    effective, non-secret configuration (incl. context size)
GET  /profiles  list of already extracted profiles
POST /extract   extract a character card + person profile from story text
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import core
from backend import BackendError, OobaboogaClient, render_prompt
from config import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("character-extractor")

settings = Settings.from_env()
client = OobaboogaClient(
    settings.api_base_url,
    model=settings.model,
    timeout=settings.request_timeout,
)

app = FastAPI(
    title="ice_ki character extractor",
    version="0.1.0",
    description=(
        "Extracts SillyTavern V2 character cards and structured person "
        "profiles from adult, fictional story text. No content moderation is "
        "applied to the extraction itself."
    ),
)


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw story text.")
    character_name: str | None = Field(
        default=None,
        description="Character to extract. Defaults to the main character.",
    )
    file_name: str | None = Field(
        default=None,
        description="Base file name. Defaults to the extracted character name.",
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "Replace existing files. Only honoured when the deployment sets "
            "EXTRACTOR_ALLOW_OVERWRITE=true."
        ),
    )
    store_raw_answer: bool = Field(
        default=False, description="Also store the unmodified model answer."
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def config() -> dict[str, object]:
    """Effective configuration. Contains no secrets."""
    return {
        "api_base_url": settings.api_base_url,
        "model": settings.model or "(currently loaded model)",
        "max_tokens": settings.max_tokens,
        "context_size": settings.context_size,
        "context_note": (
            "The context window is bounded by the loaded model and the "
            "available VRAM; TEXTGEN_CONTEXT_SIZE is the configured maximum, "
            "not an unlimited context."
        ),
        "character_card_dir": str(settings.character_card_dir),
        "profile_dir": str(settings.profile_dir),
        "allow_overwrite": settings.allow_overwrite,
        "image_prompt_safety_mode": settings.image_prompt_safety_mode,
    }


@app.get("/profiles")
def list_profiles() -> dict[str, list[str]]:
    if not settings.profile_dir.exists():
        return {"profiles": []}
    return {"profiles": sorted(p.name for p in settings.profile_dir.glob("*.json"))}


@app.post("/extract", status_code=201)
def extract(request: ExtractRequest) -> dict[str, object]:
    story_text = request.text.strip()
    if not story_text:
        raise HTTPException(status_code=422, detail="text must not be empty")

    prompt = render_prompt(story_text, request.character_name)
    try:
        raw_answer = client.complete(prompt, max_tokens=settings.max_tokens)
        profile = core.normalize_person_profile(core.parse_model_json(raw_answer))
    except BackendError as exc:
        logger.warning("backend error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except core.ExtractionError as exc:
        logger.warning("extraction error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        base_name = core.safe_filename(request.file_name or profile["name"])
    except core.ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    allow_overwrite = settings.allow_overwrite and request.overwrite
    card = core.build_character_card(profile)
    visual_prompt = core.build_visual_prompt(
        profile, safety_mode=settings.image_prompt_safety_mode
    )
    document = {
        "profile": profile,
        "visual_prompt": visual_prompt,
        "metadata": core.build_source_metadata(
            source_name=request.character_name,
            model=settings.model,
            profile=profile,
        ),
    }

    card_path = settings.character_card_dir / f"{base_name}.json"
    profile_path = settings.profile_dir / f"{base_name}.profile.json"
    try:
        core.write_json_file(card_path, card, allow_overwrite=allow_overwrite)
        core.write_json_file(profile_path, document, allow_overwrite=allow_overwrite)
        if request.store_raw_answer:
            raw_path = settings.raw_output_dir / f"{base_name}.raw.json"
            core.write_json_file(
                raw_path, {"answer": raw_answer}, allow_overwrite=allow_overwrite
            )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        logger.error("failed to write result: %s", exc)
        raise HTTPException(status_code=500, detail="failed to write result files") from exc

    return {
        "character_card_path": str(card_path),
        "profile_path": str(profile_path),
        "confidence": document["metadata"]["confidence"],
        "visual_prompt": visual_prompt,
        "character_card": card,
        "profile": profile,
    }
