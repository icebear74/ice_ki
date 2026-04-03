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
import json
import logging
import os
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import (
    FileResponse,
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
    delete_identity,
    refresh_supervectors,
    add_voice_sample,
    list_voice_samples,
    confirm_voice_sample,
    delete_voice_sample,
    update_segment_identity,
    get_episode_segments,
    get_segment_embedding,
    list_processed_episodes,
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
    and stream the result to the browser.  Works for any source format (MKV,
    AVI, H.265, DivX, …) without requiring browser codec support.

    Uses asyncio.create_subprocess_exec so FFmpeg stdout is read asynchronously
    and never causes a pipe-buffer deadlock.  stderr is discarded (DEVNULL) so
    FFmpeg cannot block on its own error output either.

    ?t=<seconds>  optional start offset – FFmpeg seeks to this position before
                  encoding so the browser always receives a playable stream from
                  the very first byte.
    """
    candidate = _resolve_video_path(path)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if t > 0:
        cmd += ["-ss", f"{t:.3f}"]
    cmd += [
        "-i", str(candidate),
        # Video: H.264 baseline, fast encode, good browser compatibility
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-profile:v", "baseline", "-level", "3.1",
        # Audio: AAC stereo
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        # Fragmented MP4: browser can start decoding before full file is received
        "-movflags", "frag_keyframe+empty_moov+default_base_moof",
        "-f", "mp4",
        "pipe:1",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,  # discard – never blocks FFmpeg
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

    FastAPI's FileResponse honours HTTP Range requests automatically, so the
    browser can seek to any position without re-downloading the whole file.
    Returns 404 if no preview has been generated yet.
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
