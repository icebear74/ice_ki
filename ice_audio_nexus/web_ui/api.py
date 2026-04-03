"""
ice_audio_nexus – web_ui/api.py
FastAPI backend providing:
  • Video streaming via FFmpeg (CUDA)
  • Episode segment data (JSON)
  • Identity & voice_sample management
  • Actor / Role / Production / VoiceCasting CRUD
  • Image upload (Pillow → optimised JPEG ≤ 800 px)
  • Image serving from MariaDB BLOB
  • TTS audio snippet extraction on segment assignment
  • Supervector revert endpoint
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, Body, UploadFile, File
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

load_dotenv()

# Ensure the project root is on sys.path when running via uvicorn from web_ui/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from db.database import (
    ensure_schema,
    get_connection,
    list_identities,
    get_identity,
    create_identity,
    update_identity,
    delete_identity,
    refresh_supervectors,
    revert_supervectors,
    # Named supervector groups
    list_supervector_groups,
    list_free_samples,
    create_named_supervector,
    revert_supervector_group,
    add_voice_sample,
    list_voice_samples,
    confirm_voice_sample,
    delete_voice_sample,
    update_segment_identity,
    update_segment_tts_path,
    get_episode_segments,
    get_segment_embedding,
    get_segment,
    list_processed_episodes,
    # Actor CRUD
    list_actors,
    get_actor,
    create_actor,
    update_actor,
    update_actor_image,
    get_actor_image,
    delete_actor,
    # Role CRUD
    list_roles,
    get_role,
    create_role,
    update_role,
    update_role_image,
    get_role_image,
    delete_role,
    # Production CRUD
    list_productions,
    get_production,
    create_production,
    update_production,
    delete_production,
    # Voice casting CRUD
    list_voice_castings,
    create_voice_casting,
    delete_voice_casting,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _probe_nvenc() -> bool:
    """Return True if the installed FFmpeg was compiled with the h264_nvenc encoder."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        return "h264_nvenc" in r.stdout
    except Exception:
        return False


# Probed once during app startup (lifespan) and cached here.
_NVENC_AVAILABLE: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _NVENC_AVAILABLE
    try:
        ensure_schema()
        logger.info("DB schema verified.")
    except Exception as exc:
        logger.error("DB init failed: %s", exc)
    _NVENC_AVAILABLE = _probe_nvenc()
    logger.info("NVENC available: %s", _NVENC_AVAILABLE)
    yield


app = FastAPI(title="ice_audio_nexus", version="2.0.0", lifespan=lifespan)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

VIDEO_DIR = Path(os.getenv("VIDEO_DIR", "/data/videos"))

# TTS dataset output root
TTS_DATASET_DIR = Path(os.getenv("TTS_DATASET_DIR", "data/voice_datasets"))

# Compiled once at module load – used by api_library()
_SEASON_RE = re.compile(r"^(S\d+)E\d+", re.IGNORECASE)

# Mount VIDEO_DIR for direct static access (fallback alongside /stream)
try:
    if VIDEO_DIR.exists():
        app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")
except Exception as _e:
    pass  # VIDEO_DIR not available at startup – /stream endpoint still works


# ---------------------------------------------------------------------------
# Image processing helper (Pillow → JPEG)
# ---------------------------------------------------------------------------

def _process_image_to_jpeg(data: bytes, max_side: int = 800) -> bytes:
    """Convert *data* to an optimised JPEG with max dimension *max_side* px."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Pillow is not installed. Run setup_env.sh to install it.",
        ) from exc

    img = Image.open(io.BytesIO(data))
    # Convert any mode to RGB so JPEG encoding always works (no alpha channel)
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Down-scale if either dimension exceeds max_side (preserve aspect ratio)
    w, h = img.size
    if max(w, h) > max_side:
        if w >= h:
            new_w, new_h = max_side, int(h * max_side / w)
        else:
            new_w, new_h = int(w * max_side / h), max_side
        img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Web UI entry
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    conn = get_connection()
    try:
        identities = list_identities(conn)
    finally:
        conn.close()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"identities": identities},
    )


# ---------------------------------------------------------------------------
# Identity API
# ---------------------------------------------------------------------------

@app.get("/api/identities")
def api_list_identities() -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_identities(conn))
    finally:
        conn.close()


@app.post("/api/identities")
def api_create_identity(
    name: Annotated[str, Body()],
    description: Annotated[str, Body()] = "",
    voice_actor_id: Annotated[int | None, Body()] = None,
) -> JSONResponse:
    conn = get_connection()
    try:
        new_id = create_identity(conn, name, description, voice_actor_id)
        return JSONResponse({"id": new_id, "name": name, "description": description,
                             "voice_actor_id": voice_actor_id})
    finally:
        conn.close()


@app.put("/api/identities/{identity_id}")
def api_update_identity(
    identity_id: int,
    name: Annotated[str, Body()],
    description: Annotated[str, Body()] = "",
    voice_actor_id: Annotated[int | None, Body()] = None,
) -> JSONResponse:
    conn = get_connection()
    try:
        if get_identity(conn, identity_id) is None:
            raise HTTPException(status_code=404, detail="Identity not found")
        update_identity(conn, identity_id, name, description, voice_actor_id)
        return JSONResponse({"status": "ok"})
    finally:
        conn.close()


@app.delete("/api/identities/{identity_id}")
def api_delete_identity(identity_id: int) -> JSONResponse:
    """Delete an identity and all its voice samples."""
    conn = get_connection()
    try:
        if get_identity(conn, identity_id) is None:
            raise HTTPException(status_code=404, detail="Identity not found")
        delete_identity(conn, identity_id)
        return JSONResponse({"status": "ok"})
    finally:
        conn.close()


@app.post("/api/refresh_supervectors")
def api_refresh_supervectors() -> JSONResponse:
    """
    Calculate the mean embedding (supervector) for every identity from all its
    real voice samples, store it with context='SUPERVECTOR', and mark all other
    samples as inactive so the scanner only uses supervectors for matching.
    """
    conn = get_connection()
    try:
        summary = refresh_supervectors(conn)
        return JSONResponse({"status": "ok", "updated": summary})
    except Exception as exc:
        logger.error("refresh_supervectors failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate supervectors. Check server logs.")
    finally:
        conn.close()


@app.post("/api/identities/{identity_id}/revert_supervector")
def api_revert_supervector(identity_id: int) -> JSONResponse:
    """
    Delete the supervector for *identity_id* and reactivate all original
    samples so the user can review and refine them before re-merging.
    """
    conn = get_connection()
    try:
        if get_identity(conn, identity_id) is None:
            raise HTTPException(status_code=404, detail="Identity not found")
        reactivated = revert_supervectors(conn, identity_id)
        return JSONResponse({"status": "ok", "reactivated_samples": reactivated})
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("revert_supervectors failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to revert supervector.")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Named supervector group API
# ---------------------------------------------------------------------------

@app.get("/api/identities/{identity_id}/supervector_groups")
def api_list_supervector_groups(identity_id: int) -> JSONResponse:
    """List all named supervector groups for an identity."""
    conn = get_connection()
    try:
        if get_identity(conn, identity_id) is None:
            raise HTTPException(status_code=404, detail="Identity not found")
        groups = list_supervector_groups(conn, identity_id)
        return JSONResponse(groups)
    finally:
        conn.close()


@app.get("/api/identities/{identity_id}/free_samples")
def api_list_free_samples(identity_id: int) -> JSONResponse:
    """List active raw samples for an identity that are not yet in any supervector group."""
    conn = get_connection()
    try:
        if get_identity(conn, identity_id) is None:
            raise HTTPException(status_code=404, detail="Identity not found")
        samples = list_free_samples(conn, identity_id)
        # Strip embeddings – browser only needs metadata + id for selection
        for s in samples:
            s.pop("embedding", None)
        return JSONResponse(samples)
    finally:
        conn.close()


@app.post("/api/identities/{identity_id}/supervector_groups")
def api_create_supervector_group(identity_id: int, data: dict = Body(...)) -> JSONResponse:
    """
    Create a named supervector group from a selected list of sample IDs.

    Body: { name: str, sample_ids: [int, ...] }
    """
    conn = get_connection()
    try:
        if get_identity(conn, identity_id) is None:
            raise HTTPException(status_code=404, detail="Identity not found")
        name       = str(data.get("name", "")).strip()
        sample_ids = [int(i) for i in data.get("sample_ids", [])]
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        if not sample_ids:
            raise HTTPException(status_code=400, detail="sample_ids must not be empty")
        try:
            group_id = create_named_supervector(conn, identity_id, name, sample_ids)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"status": "ok", "group_id": group_id,
                             "sample_count": len(sample_ids)})
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("create_named_supervector failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create supervector group.")
    finally:
        conn.close()


@app.delete("/api/supervector_groups/{group_id}")
def api_revert_supervector_group(group_id: int) -> JSONResponse:
    """Delete a named supervector group and reactivate all its source samples."""
    conn = get_connection()
    try:
        reactivated = revert_supervector_group(conn, group_id)
        return JSONResponse({"status": "ok", "reactivated_samples": reactivated})
    except Exception as exc:
        logger.error("revert_supervector_group failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to revert supervector group.")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Voice-sample API
# ---------------------------------------------------------------------------

@app.get("/api/identities/{identity_id}/samples")
def api_list_samples(identity_id: int) -> JSONResponse:
    conn = get_connection()
    try:
        samples = list_voice_samples(conn, identity_id)
        # Don't send raw float arrays to the browser – send metadata only
        for s in samples:
            s.pop("embedding", None)
        return JSONResponse(samples)
    finally:
        conn.close()


@app.post("/api/identities/{identity_id}/samples")
def api_add_sample(
    identity_id: int,
    data: dict = Body(...),
) -> JSONResponse:
    """
    Assign a new voice embedding (coming from an episode segment) to an
    existing identity.  Body: {segment_id, context, confirm}
    """
    conn = get_connection()
    try:
        if get_identity(conn, identity_id) is None:
            raise HTTPException(status_code=404, detail="Identity not found")

        segment_id = data.get("segment_id")
        context    = data.get("context", "")
        confirm    = bool(data.get("confirm", False))

        embedding: list[float] = data.get("embedding", [])
        if not embedding or len(embedding) != 512:
            raise HTTPException(
                status_code=400,
                detail="embedding must be a list of 512 floats",
            )

        sample_id = add_voice_sample(conn, identity_id, embedding, context, confirm)

        if segment_id is not None:
            update_segment_identity(
                conn,
                segment_id=int(segment_id),
                identity_id=identity_id,
                matched_sample_id=sample_id,
                is_suggestion=False,
            )

        return JSONResponse({"status": "ok", "sample_id": sample_id})
    finally:
        conn.close()


@app.post("/api/samples/{sample_id}/confirm")
def api_confirm_sample(sample_id: int) -> JSONResponse:
    conn = get_connection()
    try:
        confirm_voice_sample(conn, sample_id)
        return JSONResponse({"status": "ok"})
    finally:
        conn.close()


@app.delete("/api/samples/{sample_id}")
def api_delete_sample(sample_id: int) -> JSONResponse:
    conn = get_connection()
    try:
        delete_voice_sample(conn, sample_id)
        return JSONResponse({"status": "ok"})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Episode segment API
# ---------------------------------------------------------------------------

@app.get("/api/segments")
def api_segments(series: str, episode: str) -> JSONResponse:
    conn = get_connection()
    try:
        segs = get_episode_segments(conn, series, episode)
        return JSONResponse(segs)
    finally:
        conn.close()


@app.post("/api/segments/{segment_id}/assign")
def api_assign_segment(
    segment_id: int,
    data: dict = Body(...),
) -> JSONResponse:
    """
    Assign a segment to an existing identity (or create a new one).

    Body:
      identity_id  – existing identity (int) OR
      new_name     – if identity_id is omitted, create a new identity first
      context      – optional context string for the new voice_sample
      add_sample   – bool; if True, extract embedding from segment and store
                     it as a new voice_sample for the identity (multi-vector)
      extract_tts  – bool; if True (and add_sample is True), extract a clean
                     16-kHz mono WAV snippet for TTS dataset creation
    """
    conn = get_connection()
    try:
        identity_id = data.get("identity_id")
        new_name    = data.get("new_name", "").strip()

        if identity_id is None and new_name:
            identity_id = create_identity(conn, new_name, data.get("description", ""))
        elif identity_id is None:
            raise HTTPException(
                status_code=400,
                detail="Provide identity_id or new_name",
            )

        if get_identity(conn, identity_id) is None:
            raise HTTPException(status_code=404, detail="Identity not found")

        context     = data.get("context", "")
        add_sample  = bool(data.get("add_sample", False))
        extract_tts = bool(data.get("extract_tts", False))
        embedding   = data.get("embedding", [])

        # If the caller didn't supply an embedding, try to load the one that
        # the scanner persisted in episode_segments for this segment.
        if add_sample and (not embedding or len(embedding) != 512):
            embedding = get_segment_embedding(conn, segment_id) or []

        new_sample_id = None
        if add_sample and embedding and len(embedding) == 512:
            new_sample_id = add_voice_sample(
                conn, identity_id, embedding, context, is_confirmed=True
            )

        update_segment_identity(
            conn,
            segment_id=segment_id,
            identity_id=identity_id,
            matched_sample_id=new_sample_id,
            is_suggestion=False,
        )

        # TTS extraction: extract a clean WAV snippet for dataset building
        tts_path = None
        if extract_tts and add_sample and new_sample_id is not None:
            seg = get_segment(conn, segment_id)
            if seg and seg.get("video_path"):
                try:
                    tts_path = _extract_tts_snippet(
                        conn=conn,
                        segment_id=segment_id,
                        video_path=seg["video_path"],
                        start_ms=seg["start_ms"],
                        end_ms=seg["end_ms"],
                        identity_id=identity_id,
                        context=context,
                    )
                except Exception as exc:
                    logger.warning("TTS extraction failed (non-fatal): %s", exc)

        return JSONResponse({"status": "ok", "identity_id": identity_id,
                             "sample_id": new_sample_id, "tts_wav_path": tts_path})
    finally:
        conn.close()


def _extract_tts_snippet(
    conn,
    segment_id: int,
    video_path: str,
    start_ms: int,
    end_ms: int,
    identity_id: int,
    context: str,
) -> str | None:
    """
    Extract a 16-kHz mono WAV clip for TTS dataset building.

    Output path: TTS_DATASET_DIR / <voice_actor_name> / <context>_<start_ms>.wav
    The directory is created automatically.
    Returns the absolute path to the extracted file, or None on error.
    """
    from db.database import get_identity

    identity = get_identity(conn, identity_id)
    if identity is None:
        return None

    # Validate the video path via the same allowlist used by the streaming
    # endpoints – this prevents command-line injection from a manipulated DB value.
    try:
        safe_video_path = str(_resolve_video_path(video_path))
    except HTTPException:
        logger.warning("TTS extraction skipped – video path not allowed: %s", video_path)
        return None

    # Use identity name (sanitised) as folder name
    actor_name = re.sub(r'[^\w\-_ ]', '_', identity["name"]).strip()
    ctx_part   = re.sub(r'[^\w\-_ ]', '_', context or "clip").strip() or "clip"
    filename   = f"{ctx_part}_{start_ms}.wav"

    out_dir = TTS_DATASET_DIR / actor_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Resolve to absolute path and confirm it stays inside TTS_DATASET_DIR
    # (guards against path-traversal if sanitisation above were ever bypassed).
    out_path = os.path.realpath(out_dir / filename)
    if not out_path.startswith(os.path.realpath(TTS_DATASET_DIR)):
        logger.warning("TTS path escapes dataset dir – skipping: %s", out_path)
        return None

    start_s = start_ms / 1000.0
    end_s   = end_ms   / 1000.0
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-to", f"{end_s:.3f}",
        "-i", safe_video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60, shell=False)
    if result.returncode != 0:
        logger.warning("TTS FFmpeg failed: %s", result.stderr.decode(errors="replace")[-300:])
        return None

    # Store the path back to the segment row
    update_segment_tts_path(conn, segment_id, out_path)
    logger.info("TTS snippet saved → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Actor API
# ---------------------------------------------------------------------------

@app.get("/api/actors")
def api_list_actors() -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_actors(conn))
    finally:
        conn.close()


@app.post("/api/actors")
def api_create_actor(
    name: Annotated[str, Body()],
    description: Annotated[str, Body()] = "",
) -> JSONResponse:
    conn = get_connection()
    try:
        new_id = create_actor(conn, name, description)
        return JSONResponse({"id": new_id, "name": name, "description": description})
    finally:
        conn.close()


@app.put("/api/actors/{actor_id}")
def api_update_actor(
    actor_id: int,
    name: Annotated[str, Body()],
    description: Annotated[str, Body()] = "",
) -> JSONResponse:
    conn = get_connection()
    try:
        if get_actor(conn, actor_id) is None:
            raise HTTPException(status_code=404, detail="Actor not found")
        update_actor(conn, actor_id, name, description)
        return JSONResponse({"status": "ok"})
    finally:
        conn.close()


@app.delete("/api/actors/{actor_id}")
def api_delete_actor(actor_id: int) -> JSONResponse:
    conn = get_connection()
    try:
        if get_actor(conn, actor_id) is None:
            raise HTTPException(status_code=404, detail="Actor not found")
        delete_actor(conn, actor_id)
        return JSONResponse({"status": "ok"})
    finally:
        conn.close()


@app.post("/api/actors/{actor_id}/image")
async def api_upload_actor_image(actor_id: int, file: UploadFile = File(...)) -> JSONResponse:
    """Upload a profile image for an actor; converts to optimised JPEG internally."""
    conn = get_connection()
    try:
        if get_actor(conn, actor_id) is None:
            raise HTTPException(status_code=404, detail="Actor not found")
        raw = await file.read()
        jpeg_bytes = _process_image_to_jpeg(raw)
        update_actor_image(conn, actor_id, jpeg_bytes, "image/jpeg")
        return JSONResponse({"status": "ok", "size_bytes": len(jpeg_bytes)})
    finally:
        conn.close()


@app.get("/api/actors/{actor_id}/image")
def api_get_actor_image(actor_id: int) -> Response:
    """Serve the actor's profile image from the DB BLOB."""
    conn = get_connection()
    try:
        result = get_actor_image(conn, actor_id)
        if result is None:
            raise HTTPException(status_code=404, detail="No image for this actor")
        image_bytes, mime = result
        return Response(content=image_bytes, media_type=mime,
                        headers={"Cache-Control": "max-age=3600"})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Role API
# ---------------------------------------------------------------------------

@app.get("/api/roles")
def api_list_roles() -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_roles(conn))
    finally:
        conn.close()


@app.post("/api/roles")
def api_create_role(
    name: Annotated[str, Body()],
    description: Annotated[str, Body()] = "",
) -> JSONResponse:
    conn = get_connection()
    try:
        new_id = create_role(conn, name, description)
        return JSONResponse({"id": new_id, "name": name, "description": description})
    finally:
        conn.close()


@app.put("/api/roles/{role_id}")
def api_update_role(
    role_id: int,
    name: Annotated[str, Body()],
    description: Annotated[str, Body()] = "",
) -> JSONResponse:
    conn = get_connection()
    try:
        if get_role(conn, role_id) is None:
            raise HTTPException(status_code=404, detail="Role not found")
        update_role(conn, role_id, name, description)
        return JSONResponse({"status": "ok"})
    finally:
        conn.close()


@app.delete("/api/roles/{role_id}")
def api_delete_role(role_id: int) -> JSONResponse:
    conn = get_connection()
    try:
        if get_role(conn, role_id) is None:
            raise HTTPException(status_code=404, detail="Role not found")
        delete_role(conn, role_id)
        return JSONResponse({"status": "ok"})
    finally:
        conn.close()


@app.post("/api/roles/{role_id}/image")
async def api_upload_role_image(role_id: int, file: UploadFile = File(...)) -> JSONResponse:
    """Upload a character image for a role; converts to optimised JPEG internally."""
    conn = get_connection()
    try:
        if get_role(conn, role_id) is None:
            raise HTTPException(status_code=404, detail="Role not found")
        raw = await file.read()
        jpeg_bytes = _process_image_to_jpeg(raw)
        update_role_image(conn, role_id, jpeg_bytes, "image/jpeg")
        return JSONResponse({"status": "ok", "size_bytes": len(jpeg_bytes)})
    finally:
        conn.close()


@app.get("/api/roles/{role_id}/image")
def api_get_role_image(role_id: int) -> Response:
    """Serve the role's character image from the DB BLOB."""
    conn = get_connection()
    try:
        result = get_role_image(conn, role_id)
        if result is None:
            raise HTTPException(status_code=404, detail="No image for this role")
        image_bytes, mime = result
        return Response(content=image_bytes, media_type=mime,
                        headers={"Cache-Control": "max-age=3600"})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Production API
# ---------------------------------------------------------------------------

@app.get("/api/productions")
def api_list_productions() -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_productions(conn))
    finally:
        conn.close()


@app.post("/api/productions")
def api_create_production(data: dict = Body(...)) -> JSONResponse:
    conn = get_connection()
    try:
        title   = data.get("title", "").strip()
        year    = data.get("year")
        ptype   = data.get("type", "Series")
        if not title:
            raise HTTPException(status_code=400, detail="title is required")
        new_id = create_production(conn, title, year, ptype)
        return JSONResponse({"id": new_id, "title": title, "year": year, "type": ptype})
    finally:
        conn.close()


@app.put("/api/productions/{production_id}")
def api_update_production(production_id: int, data: dict = Body(...)) -> JSONResponse:
    conn = get_connection()
    try:
        if get_production(conn, production_id) is None:
            raise HTTPException(status_code=404, detail="Production not found")
        title = data.get("title", "").strip()
        year  = data.get("year")
        ptype = data.get("type", "Series")
        if not title:
            raise HTTPException(status_code=400, detail="title is required")
        update_production(conn, production_id, title, year, ptype)
        return JSONResponse({"status": "ok"})
    finally:
        conn.close()


@app.delete("/api/productions/{production_id}")
def api_delete_production(production_id: int) -> JSONResponse:
    conn = get_connection()
    try:
        if get_production(conn, production_id) is None:
            raise HTTPException(status_code=404, detail="Production not found")
        delete_production(conn, production_id)
        return JSONResponse({"status": "ok"})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Voice Casting API
# ---------------------------------------------------------------------------

@app.get("/api/voice_castings")
def api_list_voice_castings(production_id: int | None = None) -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_voice_castings(conn, production_id))
    finally:
        conn.close()


@app.post("/api/voice_castings")
def api_create_voice_casting(data: dict = Body(...)) -> JSONResponse:
    conn = get_connection()
    try:
        new_id = create_voice_casting(
            conn,
            production_id=int(data["production_id"]),
            role_id=int(data["role_id"]),
            actor_id=int(data["actor_id"]),
            voice_actor_id=int(data["voice_actor_id"]),
            language=data.get("language", "de"),
        )
        return JSONResponse({"id": new_id, "status": "ok"})
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing field: {exc}") from exc
    finally:
        conn.close()


@app.delete("/api/voice_castings/{casting_id}")
def api_delete_voice_casting(casting_id: int) -> JSONResponse:
    conn = get_connection()
    try:
        delete_voice_casting(conn, casting_id)
        return JSONResponse({"status": "ok"})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Library (DB-driven) & streaming
# ---------------------------------------------------------------------------

@app.get("/api/library")
def api_library() -> JSONResponse:
    """
    Return all episodes that have been processed by the scanner, grouped by
    series → season.  The video_path stored in episode_segments is used
    directly – no filesystem scan required.

    Response shape:
      [ { name: str,
          seasons: [ { name: str,
                       episodes: [ { title, video_path, segment_count } ] } ] } ]
    """
    conn = get_connection()
    try:
        episodes = list_processed_episodes(conn)
    finally:
        conn.close()

    # Group: series_name → season (extracted from title e.g. S01E03 → S01) → episodes
    series_map: dict[str, dict[str, list]] = {}
    for ep in episodes:
        s = ep["series_name"]
        t = ep["episode_title"]
        m = _SEASON_RE.match(t)
        season = m.group(1).upper() if m else "—"
        series_map.setdefault(s, {}).setdefault(season, []).append({
            "title":         t,
            "video_path":    ep["video_path"],
            "segment_count": ep["segment_count"],
        })

    library = [
        {
            "name": series_name,
            "seasons": [
                {"name": k, "episodes": series_map[series_name][k]}
                for k in sorted(series_map[series_name])
            ],
        }
        for series_name in sorted(series_map)
    ]
    return JSONResponse(library)


# Additional allowed root directories for streaming (space-separated env var).
_STREAM_ROOTS: list[Path] = [VIDEO_DIR.resolve()]
for _extra in os.getenv("STREAM_ALLOWED_ROOTS", "").split():
    _p = Path(_extra).resolve()
    if _p not in _STREAM_ROOTS:
        _STREAM_ROOTS.append(_p)

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm", ".ts"}


def _resolve_video_path(path: str) -> Path:
    """Resolve, validate and return the absolute Path for *path*.

    Raises HTTPException (403/404) on any security or existence failure.
    """
    path_obj = Path(path)
    candidate = path_obj.resolve() if path_obj.is_absolute() \
        else (VIDEO_DIR.resolve() / path).resolve()

    if candidate.suffix.lower() not in _VIDEO_EXTENSIONS:
        raise HTTPException(status_code=403, detail="File type not allowed")

    allowed = any(
        candidate == root or candidate.is_relative_to(root)
        for root in _STREAM_ROOTS
    )
    if not allowed and path_obj.is_absolute() and candidate.is_file():
        _STREAM_ROOTS.append(candidate.parent.resolve())
        allowed = True

    if not allowed:
        raise HTTPException(status_code=403, detail="Access to this path is not allowed")

    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return candidate


@app.get("/stream")
async def stream_video(path: str, t: float = 0.0) -> StreamingResponse:
    """
    Transcode a video through FFmpeg to H.264/AAC in a fragmented MP4 container
    and stream the result to the browser.
    """
    candidate = _resolve_video_path(path)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if t > 0:
        cmd += ["-ss", f"{t:.3f}"]
    if _NVENC_AVAILABLE:
        _video_enc = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
    else:
        _video_enc = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]
    cmd += [
        "-i", str(candidate),
        *_video_enc,
        "-profile:v", "baseline", "-level", "3.1",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        "-movflags", "frag_keyframe+empty_moov+default_base_moof",
        "-f", "mp4",
        "pipe:1",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="FFmpeg not found on this server")

    async def _iter():
        assert proc.stdout
        try:
            while True:
                chunk = await proc.stdout.read(65536)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()

    return StreamingResponse(
        _iter(),
        media_type="video/mp4",
        headers={"Cache-Control": "no-cache"},
    )


def _web_preview_path(video_path: str) -> Path | None:
    """Return the `.web.mp4` sibling of *video_path* if it exists, else None."""
    preview = Path(os.path.splitext(video_path)[0] + ".web.mp4")
    return preview if preview.exists() else None


@app.get("/api/has_preview")
def api_has_preview(path: str) -> JSONResponse:
    """Return whether a pre-transcoded .web.mp4 preview exists for *path*."""
    try:
        candidate = _resolve_video_path(path)
    except HTTPException:
        return JSONResponse({"has_preview": False})
    preview = _web_preview_path(str(candidate))
    return JSONResponse({"has_preview": preview is not None})


@app.get("/video")
def serve_video_preview(path: str) -> FileResponse:
    """
    Serve the pre-transcoded .web.mp4 preview file for *path*.
    FastAPI's FileResponse honours HTTP Range requests automatically.
    """
    candidate = _resolve_video_path(path)
    preview = _web_preview_path(str(candidate))
    if preview is None:
        raise HTTPException(
            status_code=404,
            detail="No web preview available for this file. Run the scanner first.",
        )
    return FileResponse(
        str(preview),
        media_type="video/mp4",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/probe")
def api_probe(path: str) -> JSONResponse:
    """Return the total duration (seconds) of a video file via ffprobe."""
    candidate = _resolve_video_path(path)

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(candidate),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
    except Exception as exc:
        logger.warning("ffprobe failed for %s: %s", candidate, exc)
        duration = 0.0

    return JSONResponse({"duration": duration})
