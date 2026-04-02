"""
ice_audio_nexus – web_ui/api.py
FastAPI backend providing:
  • Video streaming via FFmpeg (CUDA)
  • Episode segment data (JSON)
  • Identity & voice_sample management
  • Assign vector to existing identity (Multi-Vector Identity system)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
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
    add_voice_sample,
    list_voice_samples,
    confirm_voice_sample,
    delete_voice_sample,
    update_segment_identity,
    get_episode_segments,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ensure_schema()
        logger.info("DB schema verified.")
    except Exception as exc:
        logger.error("DB init failed: %s", exc)
    yield


app = FastAPI(title="ice_audio_nexus", version="1.0.0", lifespan=lifespan)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

VIDEO_DIR = Path(os.getenv("VIDEO_DIR", "/data/videos"))


# ---------------------------------------------------------------------------
# Startup (removed – replaced by lifespan context manager above)
# ---------------------------------------------------------------------------


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
        "index.html",
        {"request": request, "identities": identities},
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
) -> JSONResponse:
    conn = get_connection()
    try:
        new_id = create_identity(conn, name, description)
        return JSONResponse({"id": new_id, "name": name, "description": description})
    finally:
        conn.close()


@app.put("/api/identities/{identity_id}")
def api_update_identity(
    identity_id: int,
    name: Annotated[str, Body()],
    description: Annotated[str, Body()] = "",
) -> JSONResponse:
    conn = get_connection()
    try:
        if get_identity(conn, identity_id) is None:
            raise HTTPException(status_code=404, detail="Identity not found")
        update_identity(conn, identity_id, name, description)
        return JSONResponse({"status": "ok"})
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

        # Load the embedding from the segment's matched_sample or re-extract
        # For now we accept a direct float list under "embedding" for flexibility.
        embedding: list[float] = data.get("embedding", [])
        if not embedding or len(embedding) != 512:
            raise HTTPException(
                status_code=400,
                detail="embedding must be a list of 512 floats",
            )

        sample_id = add_voice_sample(conn, identity_id, embedding, context, confirm)

        # If a segment_id was supplied, update that segment's identity link
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
        embedding   = data.get("embedding", [])

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
        return JSONResponse({"status": "ok", "identity_id": identity_id,
                             "sample_id": new_sample_id})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Video listing & streaming
# ---------------------------------------------------------------------------

@app.get("/api/videos")
def api_list_videos() -> JSONResponse:
    if not VIDEO_DIR.exists():
        return JSONResponse([])
    extensions = {".mkv", ".mp4", ".avi", ".mov", ".m4v"}
    videos = [
        {"name": f.name, "path": str(f)}
        for f in sorted(VIDEO_DIR.rglob("*"))
        if f.suffix.lower() in extensions
    ]
    return JSONResponse(videos)


@app.get("/stream")
async def stream_video(
    request: Request,
    path: str,
    seek: float = 0.0,
) -> StreamingResponse:
    """
    Stream a video via FFmpeg (CUDA) as HLS-compatible H.264/AAC in fragmented
    MP4 format, which is playable by all modern browsers.
    If *seek* is given the stream starts at that second offset.

    The resolved path must be inside VIDEO_DIR to prevent path traversal.
    """
    video_path = Path(path).resolve()

    # Security: ensure the resolved path is inside the configured VIDEO_DIR
    try:
        video_path.relative_to(VIDEO_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access to this path is not allowed")

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-ss", str(seek),
        "-i", str(video_path),
        "-c:v", "h264_nvenc",       # NVIDIA hardware encoder
        "-preset", "p4",
        "-c:a", "aac",
        "-b:a", "192k",
        "-f", "mp4",
        "-movflags", "frag_keyframe+empty_moov+faststart",
        "pipe:1",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    async def _generator():
        assert proc.stdout is not None
        try:
            while True:
                chunk = await proc.stdout.read(65536)
                if not chunk:
                    break
                yield chunk
        finally:
            if proc.returncode is None:
                proc.terminate()

    return StreamingResponse(
        _generator(),
        media_type="video/mp4",
        headers={"Cache-Control": "no-cache"},
    )
