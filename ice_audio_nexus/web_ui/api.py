"""
FastAPI backend for Step-1 visual person discovery review.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from db.database import (  # noqa: E402
    assign_track,
    create_actor,
    create_role,
    ensure_schema,
    get_connection,
    get_video,
    get_track,
    list_actors,
    list_face_samples,
    list_library,
    list_overlay_events,
    list_productions,
    list_roles,
    list_video_tracks,
    list_videos,
    rematch_tracks,
    update_track_status,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

VIDEO_DIR = Path(os.getenv("VIDEO_DIR", "/data/videos")).resolve()
FACE_DATA_DIR = Path(os.getenv("FACE_DATA_DIR", "data/faces")).resolve()
_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _probe_nvenc() -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return "h264_nvenc" in r.stdout
    except Exception:
        return False


_NVENC_AVAILABLE = False
_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm", ".ts"}
_STREAM_ROOTS: list[Path] = [VIDEO_DIR]

for extra in os.getenv("STREAM_ALLOWED_ROOTS", "").split():
    p = Path(extra).resolve()
    if p not in _STREAM_ROOTS:
        _STREAM_ROOTS.append(p)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _NVENC_AVAILABLE
    ensure_schema()
    _NVENC_AVAILABLE = _probe_nvenc()
    FACE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("NVENC available: %s", _NVENC_AVAILABLE)
    yield


app = FastAPI(title="ice_audio_nexus", version="3.0.0", lifespan=lifespan)

if VIDEO_DIR.exists():
    app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")
if FACE_DATA_DIR.exists():
    app.mount("/faces", StaticFiles(directory=str(FACE_DATA_DIR)), name="faces")


def _resolve_video_path(path: str) -> Path:
    candidate = Path(path).resolve()
    if candidate.suffix.lower() not in _VIDEO_EXTENSIONS:
        raise HTTPException(status_code=403, detail="File type not allowed")
    allowed = any(candidate == root or candidate.is_relative_to(root) for root in _STREAM_ROOTS)
    if not allowed:
        raise HTTPException(status_code=403, detail="Access to this path is not allowed")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return candidate


def _resolve_video_id_path(video_id: int) -> Path:
    conn = get_connection()
    try:
        video = get_video(conn, video_id)
    finally:
        conn.close()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return _resolve_video_path(str(video["video_path"]))


def _probe_duration(path: str) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        data = json.loads(result.stdout or "{}")
        duration = float(data.get("format", {}).get("duration", 0) or 0)
        for stream in data.get("streams", []):
            try:
                duration = max(duration, float(stream.get("duration", 0) or 0))
            except (TypeError, ValueError):
                pass
        return max(0.0, duration)
    except Exception:
        return 0.0


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/api/library")
def api_library() -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_library(conn))
    finally:
        conn.close()


@app.get("/api/productions")
def api_productions() -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_productions(conn))
    finally:
        conn.close()


@app.get("/api/videos")
def api_videos(production_id: int | None = None) -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_videos(conn, production_id=production_id))
    finally:
        conn.close()


@app.get("/api/videos/{video_id}/tracks")
def api_video_tracks(video_id: int, clear_only: bool = False, status: str | None = None) -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_video_tracks(conn, video_id, clear_only=clear_only, status=status))
    finally:
        conn.close()


@app.get("/api/tracks/{track_id}")
def api_track(track_id: int) -> JSONResponse:
    conn = get_connection()
    try:
        track = get_track(conn, track_id)
        if not track:
            raise HTTPException(status_code=404, detail="Track not found")
        return JSONResponse(track)
    finally:
        conn.close()


@app.post("/api/tracks/{track_id}/assign")
def api_assign_track(track_id: int, payload: dict = Body(...)) -> JSONResponse:
    actor_id = payload.get("actor_id")
    new_actor_name = (payload.get("new_actor_name") or "").strip()
    role_id = payload.get("role_id")
    role_name = (payload.get("new_role_name") or "").strip()
    add_sample = bool(payload.get("add_sample", True))

    if actor_id is None and not new_actor_name:
        raise HTTPException(status_code=400, detail="Provide actor_id or new_actor_name")

    conn = get_connection()
    try:
        if actor_id is None:
            actor_id = create_actor(conn, new_actor_name, payload.get("description", ""))
        if role_id is None and role_name:
            role_id = create_role(conn, role_name, "")
        result = assign_track(
            conn,
            track_id=track_id,
            actor_id=int(actor_id),
            role_id=int(role_id) if role_id is not None else None,
            add_sample=add_sample,
            source="manual",
        )
        return JSONResponse({"ok": True, **result})
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        conn.close()


@app.post("/api/tracks/{track_id}/status")
def api_update_track_status(track_id: int, payload: dict = Body(...)) -> JSONResponse:
    status = str(payload.get("status", "")).strip()
    conn = get_connection()
    try:
        update_track_status(conn, track_id, status)
        return JSONResponse({"ok": True, "track_id": track_id, "status": status})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        conn.close()


@app.get("/api/videos/{video_id}/overlay")
def api_overlay(video_id: int) -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_overlay_events(conn, video_id))
    finally:
        conn.close()


@app.get("/api/actors")
def api_actors() -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_actors(conn))
    finally:
        conn.close()


@app.post("/api/actors")
def api_create_actor(payload: dict = Body(...)) -> JSONResponse:
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    conn = get_connection()
    try:
        actor_id = create_actor(conn, name, payload.get("description", ""))
        return JSONResponse({"id": actor_id, "name": name})
    finally:
        conn.close()


@app.get("/api/roles")
def api_roles() -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_roles(conn))
    finally:
        conn.close()


@app.post("/api/roles")
def api_create_role(payload: dict = Body(...)) -> JSONResponse:
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    conn = get_connection()
    try:
        role_id = create_role(conn, name, payload.get("description", ""))
        return JSONResponse({"id": role_id, "name": name})
    finally:
        conn.close()


@app.get("/api/face_samples")
def api_face_samples(actor_id: int | None = None) -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_face_samples(conn, actor_id=actor_id))
    finally:
        conn.close()


@app.post("/api/rematch")
def api_rematch(payload: dict = Body(default={})) -> JSONResponse:
    conn = get_connection()
    try:
        result = rematch_tracks(
            conn,
            video_id=payload.get("video_id"),
            production_id=payload.get("production_id"),
            actor_id=payload.get("actor_id"),
            assign_threshold=float(payload.get("assign_threshold", os.getenv("FACE_REMATCH_ASSIGN_THRESHOLD", "0.90"))),
            suggest_threshold=float(payload.get("suggest_threshold", os.getenv("FACE_REMATCH_SUGGEST_THRESHOLD", "0.78"))),
        )
        return JSONResponse(result)
    finally:
        conn.close()


@app.get("/api/probe/{video_id}")
def api_probe(video_id: int) -> JSONResponse:
    candidate = _resolve_video_id_path(video_id)
    return JSONResponse({"duration": _probe_duration(str(candidate))})


@app.get("/stream/{video_id}")
async def stream_video(video_id: int, t: float = 0.0) -> StreamingResponse:
    candidate = _resolve_video_id_path(video_id)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if t > 0:
        cmd += ["-ss", f"{t:.3f}"]
    if _NVENC_AVAILABLE:
        video_codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
    else:
        video_codec = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]
    cmd += [
        "-i",
        str(candidate),
        *video_codec,
        "-profile:v",
        "baseline",
        "-level",
        "3.1",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ac",
        "2",
        "-movflags",
        "frag_keyframe+empty_moov+default_base_moof",
        "-f",
        "mp4",
        "pipe:1",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="FFmpeg not found on this server") from exc

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

    return StreamingResponse(_iter(), media_type="video/mp4", headers={"Cache-Control": "no-cache"})


@app.get("/video/{video_id}")
def serve_video_file(video_id: int) -> FileResponse:
    candidate = _resolve_video_id_path(video_id)
    return FileResponse(str(candidate), media_type="video/mp4", headers={"Cache-Control": "no-cache"})
