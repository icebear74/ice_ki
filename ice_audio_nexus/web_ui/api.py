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
import re
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
    add_voice_sample,
    list_voice_samples,
    confirm_voice_sample,
    delete_voice_sample,
    update_segment_identity,
    get_episode_segments,
    list_processed_episodes,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# h264_nvenc availability is probed once at startup
_NVENC_AVAILABLE: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _NVENC_AVAILABLE
    try:
        ensure_schema()
        logger.info("DB schema verified.")
    except Exception as exc:
        logger.error("DB init failed: %s", exc)

    # Probe whether h264_nvenc is available in the installed ffmpeg
    try:
        probe = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-encoders",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await probe.communicate()
        _NVENC_AVAILABLE = b"h264_nvenc" in out
        logger.info("h264_nvenc encoder: %s", "available" if _NVENC_AVAILABLE else "NOT available – using libx264")
    except Exception as exc:
        logger.warning("Could not probe ffmpeg encoders: %s", exc)

    yield


app = FastAPI(title="ice_audio_nexus", version="1.0.0", lifespan=lifespan)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

VIDEO_DIR = Path(os.getenv("VIDEO_DIR", "/data/videos"))

# Compiled once at module load – used by api_library()
_SEASON_RE = re.compile(r"^(S\d+)E\d+", re.IGNORECASE)

# Mount VIDEO_DIR for direct static access (fallback alongside /stream)
try:
    if VIDEO_DIR.exists():
        app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")
except Exception as _e:
    pass  # VIDEO_DIR not available at startup – /stream endpoint still works


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
# The scanner stores absolute paths that may differ from VIDEO_DIR, so we
# maintain a list of permitted roots.  VIDEO_DIR is always included.
_STREAM_ROOTS: list[Path] = [VIDEO_DIR.resolve()]
for _extra in os.getenv("STREAM_ALLOWED_ROOTS", "").split():
    _p = Path(_extra).resolve()
    if _p not in _STREAM_ROOTS:
        _STREAM_ROOTS.append(_p)

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm", ".ts"}


@app.get("/stream")
async def stream_video(
    request: Request,
    path: str,
    seek: float = 0.0,
) -> StreamingResponse:
    """
    Stream a video via FFmpeg as a browser-compatible fragmented MP4.

    *path* may be:
      - an absolute path as stored in episode_segments.video_path, OR
      - a path relative to VIDEO_DIR.

    Security: the resolved path must sit inside one of the allowed roots
    (VIDEO_DIR or STREAM_ALLOWED_ROOTS) and must have a known video extension.

    Uses h264_nvenc when available (probed at startup), falls back to libx264.
    """
    path_obj = Path(path)
    candidate = path_obj.resolve() if path_obj.is_absolute() \
        else (VIDEO_DIR.resolve() / path).resolve()

    # Must have a recognised video extension
    if candidate.suffix.lower() not in _VIDEO_EXTENSIONS:
        raise HTTPException(status_code=403, detail="File type not allowed")

    # Must sit inside at least one permitted root directory.
    # When the scanner stores an absolute path on a different mount, we also
    # accept it if it is a real file (its parent is auto-added to the roots
    # so that traversal back out of that directory is still prevented).
    allowed = any(
        candidate == root or candidate.is_relative_to(root)
        for root in _STREAM_ROOTS
    )
    if not allowed and path_obj.is_absolute() and candidate.is_file():
        # Accept the scanner-stored path; add its directory as a new root
        # so future requests to siblings are also allowed without re-checking.
        _STREAM_ROOTS.append(candidate.parent.resolve())
        allowed = True

    if not allowed:
        raise HTTPException(status_code=403, detail="Access to this path is not allowed")

    safe_path = candidate
    if not safe_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    if _NVENC_AVAILABLE:
        encoder_args = ["-c:v", "h264_nvenc", "-preset", "p4"]
        hwaccel_args = ["-hwaccel", "cuda"]
    else:
        encoder_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
        hwaccel_args = []

    cmd = [
        "ffmpeg", "-y",
        *hwaccel_args,
        "-ss", str(seek),
        "-i", str(safe_path),
        *encoder_args,
        "-c:a", "aac",
        "-b:a", "192k",
        "-f", "mp4",
        # frag_keyframe+empty_moov: browser-streamable fragmented MP4 via pipe
        # (faststart intentionally omitted – requires seekable output)
        "-movflags", "frag_keyframe+empty_moov",
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
