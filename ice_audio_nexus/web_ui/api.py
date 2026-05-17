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
import threading
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
    block_group_expansion,
    clear_video_scan_data,
    cluster_tracks_into_groups,
    create_actor,
    create_role,
    create_voice_actor,
    create_visual_group,
    create_visual_seed,
    delete_persona_catalog_entry,
    delete_role_cast_assignment,
    ensure_schema,
    get_connection,
    get_video,
    get_track,
    get_visual_group,
    list_actors,
    list_face_samples,
    list_library,
    list_overlay_events,
    list_persona_catalog,
    list_productions,
    list_role_cast_assignments,
    list_roles,
    list_video_tracks,
    list_videos,
    list_voice_actors,
    list_visual_groups,
    list_visual_seeds,
    rematch_tracks,
    remove_visual_seed,
    run_expansion_for_group,
    set_video_scan_status,
    set_video_expansion_release,
    trigger_group_expansion,
    unlink_detection_from_track,
    update_track_seed_workflow,
    update_track_status,
    update_visual_group,
    upsert_role_cast_assignment,
    upsert_persona_catalog,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


VIDEO_DIR = Path(os.getenv("VIDEO_DIR", "/data/videos")).resolve()
FACE_DATA_DIR = Path(os.getenv("FACE_DATA_DIR", "data/faces")).resolve()
_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _probe_nvenc() -> bool:
    """Return True only when a real NVENC test-encode succeeds."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if "h264_nvenc" not in r.stdout:
            return False
    except Exception:
        return False

    # Encoder listed – now verify it actually works with a tiny null-source encode.
    try:
        r2 = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=black:s=64x64:r=1",
                "-t", "0.1",
                "-c:v", "h264_nvenc",
                "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if r2.returncode != 0:
            logger.warning("NVENC test-encode failed (rc=%d): %s", r2.returncode, r2.stderr.strip())
            return False
        return True
    except Exception as exc:
        logger.warning("NVENC test-encode exception: %s", exc)
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
    logger.info("NVENC available: %s", _NVENC_AVAILABLE)
    yield


# Ensure face-data directory exists at module-load time so the static mount
# always succeeds (creating it in lifespan would be too late).
FACE_DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ice_audio_nexus", version="3.0.0", lifespan=lifespan)

if VIDEO_DIR.exists():
    app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")
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


@app.post("/api/videos/{video_id}/expansion_release")
def api_video_expansion_release(video_id: int, payload: dict = Body(...)) -> JSONResponse:
    conn = get_connection()
    try:
        released = bool(payload.get("released", False))
        result = set_video_expansion_release(conn, video_id, released=released)
        return JSONResponse({"ok": True, **result})
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
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


@app.post("/api/tracks/{track_id}/workflow")
def api_update_track_workflow(track_id: int, payload: dict = Body(...)) -> JSONResponse:
    conn = get_connection()
    try:
        result = update_track_seed_workflow(
            conn,
            track_id,
            stage=payload.get("stage"),
            review_state=payload.get("review_state"),
            group_label=payload.get("group_label"),
            expansion_state=payload.get("expansion_state"),
            notes=payload.get("notes"),
        )
        return JSONResponse({"ok": True, **result})
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


@app.get("/api/voice_actors")
def api_voice_actors() -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_voice_actors(conn))
    finally:
        conn.close()


@app.post("/api/voice_actors")
def api_create_voice_actor(payload: dict = Body(...)) -> JSONResponse:
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    conn = get_connection()
    try:
        voice_actor_id = create_voice_actor(conn, name, payload.get("notes", ""))
        return JSONResponse({"id": voice_actor_id, "name": name})
    finally:
        conn.close()


@app.get("/api/face_samples")
def api_face_samples(actor_id: int | None = None) -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_face_samples(conn, actor_id=actor_id))
    finally:
        conn.close()


@app.delete("/api/detections/{detection_id}")
def api_unlink_detection(detection_id: int) -> JSONResponse:
    """Unlink a single face detection from its track (sets track_id = NULL).

    Use this to remove an outlier face crop from a cluster without deleting
    the underlying image or detection record.
    """
    conn = get_connection()
    try:
        updated = unlink_detection_from_track(conn, detection_id)
        if not updated:
            raise HTTPException(
                status_code=404,
                detail=f"Detection {detection_id} not found or not linked to any track",
            )
        return JSONResponse({"ok": True, "detection_id": detection_id})
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


# ─────────────────────── WP1+WP2+WP3 – visual groups ───────────────────────

@app.get("/api/visual_groups")
def api_visual_groups(production_id: int | None = None) -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_visual_groups(conn, production_id=production_id))
    finally:
        conn.close()


@app.get("/api/visual_groups/{group_id}")
def api_visual_group(group_id: int) -> JSONResponse:
    conn = get_connection()
    try:
        g = get_visual_group(conn, group_id)
        if not g:
            raise HTTPException(status_code=404, detail="Visual group not found")
        return JSONResponse(g)
    finally:
        conn.close()


@app.post("/api/visual_groups")
def api_create_visual_group(payload: dict = Body(...)) -> JSONResponse:
    production_id = payload.get("production_id")
    if not production_id:
        raise HTTPException(status_code=400, detail="production_id required")
    conn = get_connection()
    try:
        gid = create_visual_group(
            conn,
            production_id=int(production_id),
            label=payload.get("label"),
            notes=payload.get("notes"),
        )
        return JSONResponse({"ok": True, "group_id": gid})
    finally:
        conn.close()


@app.put("/api/visual_groups/{group_id}")
def api_update_visual_group(group_id: int, payload: dict = Body(...)) -> JSONResponse:
    conn = get_connection()
    try:
        update_kwargs: dict = {
            "label": payload.get("label"),
            "review_state": payload.get("review_state"),
            "expansion_state": payload.get("expansion_state"),
            "notes": payload.get("notes"),
        }

        new_actor_name = (payload.get("new_actor_name") or "").strip()
        if "assigned_actor_id" in payload or new_actor_name:
            actor_id = payload.get("assigned_actor_id")
            if actor_id is None and new_actor_name:
                actor_id = create_actor(conn, new_actor_name, payload.get("new_actor_description", ""))
            update_kwargs["assigned_actor_id"] = int(actor_id) if actor_id else None

        new_role_name = (payload.get("new_role_name") or "").strip()
        if "assigned_role_id" in payload or new_role_name:
            role_id = payload.get("assigned_role_id")
            if role_id is None and new_role_name:
                role_id = create_role(conn, new_role_name, "")
            update_kwargs["assigned_role_id"] = int(role_id) if role_id else None

        result = update_visual_group(
            conn,
            group_id,
            **update_kwargs,
        )
        return JSONResponse({"ok": True, **result})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        conn.close()


@app.delete("/api/visual_seeds/{seed_id}")
def api_remove_visual_seed(seed_id: int) -> JSONResponse:
    conn = get_connection()
    try:
        ok = remove_visual_seed(conn, seed_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Seed not found")
        return JSONResponse({"ok": True, "seed_id": seed_id})
    finally:
        conn.close()


# ─────────────────────────── WP2 – clustering ───────────────────────────────

@app.post("/api/productions/{production_id}/cluster")
def api_cluster(production_id: int, payload: dict = Body(default={})) -> JSONResponse:
    """WP2: Conservative clustering – groups tracks into visual_person_NNN groups."""
    threshold = float(payload.get("similarity_threshold", 0.92))
    conn = get_connection()
    try:
        result = cluster_tracks_into_groups(conn, production_id, similarity_threshold=threshold)
        return JSONResponse({"ok": True, **result})
    finally:
        conn.close()


# ─────────────────────── WP5 – expansion control ────────────────────────────

@app.post("/api/visual_groups/{group_id}/expand")
def api_trigger_expansion(group_id: int) -> JSONResponse:
    """WP5: Mark a confirmed group as ready for expansion. Blocks irrelevant groups."""
    conn = get_connection()
    try:
        result = trigger_group_expansion(conn, group_id)
        return JSONResponse(result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        conn.close()


@app.post("/api/visual_groups/{group_id}/block_expansion")
def api_block_expansion(group_id: int) -> JSONResponse:
    """WP5: Explicitly block expansion for a group."""
    conn = get_connection()
    try:
        result = block_group_expansion(conn, group_id)
        return JSONResponse(result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        conn.close()


@app.post("/api/visual_groups/{group_id}/run_expansion")
def api_run_expansion(group_id: int, payload: dict = Body(default={})) -> JSONResponse:
    """Step 1C: Run the expansion engine for a confirmed group.

    Finds all unassigned clear tracks in the same production that match the
    group's seed centroid (cosine similarity >= match_threshold) and assigns
    them to the group.  Only works for confirmed groups; irrelevant/ignored
    groups are rejected.
    """
    threshold = float(payload.get("match_threshold", os.getenv("FACE_EXPAND_THRESHOLD", "0.70")))
    conn = get_connection()
    try:
        group = get_visual_group(conn, group_id)
        if not group:
            raise HTTPException(status_code=404, detail="Visual group not found")
        production_id = group.get("production_id")
        allowed_video_ids: list[int] = []
        if production_id is not None:
            videos = list_videos(conn, production_id=int(production_id))
            allowed_video_ids = [int(v["id"]) for v in videos if bool(v.get("expansion_released"))]
        if not allowed_video_ids:
            return JSONResponse(
                {
                    "ok": False,
                    "reason": "No released episodes for this production. Release episodes before expansion.",
                    "tracks_matched": 0,
                    "seeds_added": 0,
                }
            )
        result = run_expansion_for_group(
            conn,
            group_id,
            match_threshold=threshold,
            allowed_video_ids=allowed_video_ids,
        )
        return JSONResponse(result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        conn.close()


# ─────────────────────────── WP4 – persona catalog ──────────────────────────

@app.get("/api/persona_catalog")
def api_persona_catalog(production_id: int | None = None) -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_persona_catalog(conn, production_id=production_id))
    finally:
        conn.close()


@app.post("/api/persona_catalog")
def api_upsert_persona(payload: dict = Body(...)) -> JSONResponse:
    conn = get_connection()
    try:
        role_id = payload.get("role_id")
        role_name = (payload.get("role_name") or "").strip()
        if not role_id and role_name:
            role_id = create_role(conn, role_name, "")
        actor_id = payload.get("actor_id")
        actor_name = (payload.get("actor_name") or "").strip()
        if not actor_id and actor_name:
            actor_id = create_actor(conn, actor_name, "")
        voice_actor_id = payload.get("voice_actor_id")
        voice_actor_name = (payload.get("voice_actor_name") or "").strip()
        if not voice_actor_id and voice_actor_name:
            voice_actor_id = create_voice_actor(conn, voice_actor_name, "")
        entry_id = upsert_persona_catalog(
            conn,
            production_id=int(payload["production_id"]) if payload.get("production_id") else None,
            role_id=int(role_id) if role_id is not None else None,
            actor_id=int(actor_id) if actor_id is not None else None,
            voice_actor_id=int(voice_actor_id) if voice_actor_id is not None else None,
            voice_actor_name=voice_actor_name or None,
            language=str(payload.get("language") or "de"),
            relevance=int(payload.get("relevance", 1)),
            notes=payload.get("notes"),
        )
        return JSONResponse({"ok": True, "id": entry_id})
    except (KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        conn.close()


@app.get("/api/role_cast_assignments")
def api_role_cast_assignments(production_id: int | None = None) -> JSONResponse:
    conn = get_connection()
    try:
        return JSONResponse(list_role_cast_assignments(conn, production_id=production_id))
    finally:
        conn.close()


@app.post("/api/role_cast_assignments")
def api_upsert_role_cast_assignment(payload: dict = Body(...)) -> JSONResponse:
    conn = get_connection()
    try:
        if not payload.get("production_id"):
            raise HTTPException(status_code=400, detail="production_id is required")
        role_id = payload.get("role_id")
        role_name = (payload.get("role_name") or "").strip()
        if not role_id and role_name:
            role_id = create_role(conn, role_name, "")
        actor_id = payload.get("actor_id")
        actor_name = (payload.get("actor_name") or "").strip()
        if not actor_id and actor_name:
            actor_id = create_actor(conn, actor_name, "")
        voice_actor_id = payload.get("voice_actor_id")
        voice_actor_name = (payload.get("voice_actor_name") or "").strip()
        if not voice_actor_id and voice_actor_name:
            voice_actor_id = create_voice_actor(conn, voice_actor_name, "")
        if not role_id:
            raise HTTPException(status_code=400, detail="role_id or role_name is required")
        if not voice_actor_id:
            raise HTTPException(status_code=400, detail="voice_actor_id or voice_actor_name is required")

        assignment_id = upsert_role_cast_assignment(
            conn,
            production_id=int(payload["production_id"]),
            role_id=int(role_id),
            actor_id=int(actor_id) if actor_id is not None else None,
            voice_actor_id=int(voice_actor_id),
            language=str(payload.get("language") or "de"),
            relevance=int(payload.get("relevance", 1)),
            start_season=int(payload.get("start_season", 1)),
            start_episode=int(payload.get("start_episode", 1)),
            notes=payload.get("notes"),
        )
        return JSONResponse({"ok": True, "id": assignment_id})
    finally:
        conn.close()


@app.delete("/api/role_cast_assignments/{assignment_id}")
def api_delete_role_cast_assignment(assignment_id: int) -> JSONResponse:
    conn = get_connection()
    try:
        ok = delete_role_cast_assignment(conn, assignment_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Assignment not found")
        return JSONResponse({"ok": True, "id": assignment_id})
    finally:
        conn.close()


@app.delete("/api/persona_catalog/{entry_id}")
def api_delete_persona(entry_id: int) -> JSONResponse:
    conn = get_connection()
    try:
        ok = delete_persona_catalog_entry(conn, entry_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Entry not found")
        return JSONResponse({"ok": True, "id": entry_id})
    finally:
        conn.close()


def _run_rescan_bg(video_id: int, video_path: str, scan_kwargs: dict) -> None:
    """Background thread: clear existing scan data and re-scan a video."""
    try:
        # Import scanner lazily so the API can start without cv2 installed
        from processor.scanner import scan_video  # noqa: F401  # type: ignore[import]
    except ImportError:
        logger.error("scanner not importable — missing scanner dependencies (cv2 / torch / facenet-pytorch)?")
        return

    conn = get_connection()
    try:
        set_video_scan_status(conn, video_id, "scanning")
        clear_video_scan_data(conn, video_id)
    finally:
        conn.close()

    try:
        result = scan_video(video_path, **scan_kwargs)
        logger.info("Rescan completed for video_id=%d: %s", video_id, result)
    except Exception as exc:
        logger.error("Rescan failed for video_id=%d: %s", video_id, exc)
        conn2 = get_connection()
        try:
            set_video_scan_status(conn2, video_id, "failed")
        finally:
            conn2.close()


@app.post("/api/videos/{video_id}/rescan")
def api_rescan_video(video_id: int, payload: dict = Body(default={})) -> JSONResponse:
    """Delete all scan data for this video and trigger a fresh scan in the background.

    The video's scan_status is immediately set to 'scanning'. Poll
    GET /api/videos?production_id=… or the library endpoint to track progress.
    """
    conn = get_connection()
    try:
        video = get_video(conn, video_id)
    finally:
        conn.close()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    scan_kwargs: dict = {
        "sample_fps": float(payload.get("fps", os.getenv("FACE_SCAN_FPS", "4.0"))),
        "start_offset_seconds": float(payload.get("start_offset_seconds", os.getenv("FACE_SCAN_START_OFFSET_SECONDS", "0.0"))),
        "max_sampled_frames": int(payload.get("max_sampled_frames", os.getenv("FACE_SCAN_MAX_SAMPLED_FRAMES", "0"))),
        "min_clear_seconds": float(payload.get("min_clear_seconds", os.getenv("FACE_MIN_CLEAR_SECONDS", "2.0"))),
        "min_face_area_ratio": float(payload.get("min_face_area_ratio", os.getenv("FACE_MIN_AREA_RATIO", "0.06"))),
        "min_sharpness": float(payload.get("min_sharpness", os.getenv("FACE_MIN_SHARPNESS", "70.0"))),
        "min_brightness": float(payload.get("min_brightness", os.getenv("FACE_MIN_BRIGHTNESS", "40.0"))),
        "min_quality_score": float(payload.get("min_quality_score", os.getenv("FACE_MIN_QUALITY_SCORE", "0.55"))),
        "seed_acceptance_threshold": float(payload.get("seed_acceptance_threshold", os.getenv("FACE_SEED_ACCEPTANCE_THRESHOLD", "0.60"))),
        "max_aspect_ratio_deviation": float(payload.get("max_aspect_ratio_deviation", os.getenv("FACE_MAX_ASPECT_RATIO_DEVIATION", "0.65"))),
        "min_stability": float(payload.get("min_stability", os.getenv("FACE_MIN_STABILITY", "0.45"))),
        "dnn_confidence": float(
            payload.get(
                "dnn_confidence",
                os.getenv("FACE_DETECTOR_SCORE_THRESHOLD", os.getenv("FACE_DNN_CONFIDENCE", "0.65")),
            )
        ),
        "min_face_size_px": int(payload.get("min_face_size_px", os.getenv("FACE_MIN_SIZE_PX", "80"))),
        "verifier_enabled": _as_bool(payload.get("verifier_enabled"), _as_bool(os.getenv("FACE_VERIFIER_ENABLED"), True)),
        "verifier_score_threshold": float(payload.get("verifier_score_threshold", os.getenv("FACE_VERIFIER_SCORE_THRESHOLD", "0.92"))),
        "verifier_min_area_ratio": float(payload.get("verifier_min_area_ratio", os.getenv("FACE_VERIFIER_MIN_AREA_RATIO", "0.25"))),
        "verifier_max_center_offset": float(payload.get("verifier_max_center_offset", os.getenv("FACE_VERIFIER_MAX_CENTER_OFFSET", "0.45"))),
        "prefer_gpu": _as_bool(payload.get("prefer_gpu"), _as_bool(os.getenv("FACE_GPU_ENABLED"), True)),
        "gpu_device_id": int(payload.get("gpu_device_id", os.getenv("FACE_GPU_DEVICE_ID", "0"))),
        "detector_device": payload.get("detector_device", os.getenv("FACE_DETECTOR_DEVICE")),
        "verifier_device": payload.get("verifier_device", os.getenv("FACE_VERIFIER_DEVICE")),
        "embedding_device": payload.get("embedding_device", os.getenv("FACE_EMBEDDING_DEVICE")),
        "gpu_diagnostics": _as_bool(payload.get("gpu_diagnostics"), _as_bool(os.getenv("FACE_GPU_DIAGNOSTICS"), True)),
        "duplicate_similarity_threshold": float(payload.get("duplicate_similarity_threshold", os.getenv("FACE_SEED_DUPLICATE_SIMILARITY_THRESHOLD", "0.985"))),
        "write_debug_stats": _as_bool(payload.get("write_debug_stats"), _as_bool(os.getenv("FACE_SEED_DEBUG_STATS_ENABLED"), False)),
        "debug_stats_dir": payload.get("debug_stats_dir", os.getenv("FACE_SEED_DEBUG_STATS_DIR")),
    }

    t = threading.Thread(
        target=_run_rescan_bg,
        args=(video_id, str(video["video_path"]), scan_kwargs),
        daemon=True,
        name=f"rescan-{video_id}",
    )
    t.start()
    return JSONResponse({"ok": True, "video_id": video_id, "status": "scanning"})


def _run_scan_directory_bg(directory: str, production: str | None, skip_done: bool, recursive: bool, scan_kwargs: dict) -> None:
    """Background thread: scan all video files in a directory."""
    try:
        from processor.scanner import scan_directory  # type: ignore[import]
    except ImportError:
        logger.error("scanner not importable — missing scanner dependencies (cv2 / torch / facenet-pytorch)?")
        return

    try:
        result = scan_directory(
            directory,
            production=production,
            skip_done=skip_done,
            recursive=recursive,
            **scan_kwargs,
        )
        logger.info(
            "Directory scan finished: scanned=%d skipped=%d failed=%d dir=%s",
            result["scanned"], result["skipped"], result["failed"], directory,
        )
    except Exception as exc:
        logger.error("Directory scan failed (%s): %s", directory, exc)


@app.post("/api/scan/directory")
def api_scan_directory(payload: dict = Body(...)) -> JSONResponse:
    """Start a background scan of all video files in a directory.

    Required body field:
        directory (str): absolute path to the folder to scan.

    Optional fields:
        production (str): override production name (default: folder name).
        rescan (bool): re-scan already-completed videos (default: false).
        recursive (bool): search sub-directories (default: false).
        fps, start_offset_seconds, max_sampled_frames,
        min_clear_seconds, min_face_area_ratio, min_sharpness, min_brightness,
        min_quality_score, seed_acceptance_threshold, max_aspect_ratio_deviation,
        min_stability, dnn_confidence, min_face_size_px,
        verifier_enabled, verifier_score_threshold, verifier_min_area_ratio,
        verifier_max_center_offset, duplicate_similarity_threshold,
        prefer_gpu, gpu_device_id, detector_device, verifier_device, embedding_device,
        gpu_diagnostics, write_debug_stats, debug_stats_dir
        — scanner tunables.
    """
    directory = (payload.get("directory") or "").strip()
    if not directory:
        raise HTTPException(status_code=400, detail="'directory' is required")

    root = Path(directory).resolve()
    # Restrict to allowed roots (same allowlist used by video streaming)
    allowed = any(root == ar or root.is_relative_to(ar) for ar in _STREAM_ROOTS)
    if not allowed:
        raise HTTPException(
            status_code=403,
            detail=(
                "Directory is not within an allowed root. "
                "Configure VIDEO_DIR or STREAM_ALLOWED_ROOTS to include this path."
            ),
        )
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {root}")

    production = (payload.get("production") or "").strip() or None
    skip_done = not bool(payload.get("rescan", False))
    recursive = bool(payload.get("recursive", False))

    scan_kwargs: dict = {
        "sample_fps": float(payload.get("fps", os.getenv("FACE_SCAN_FPS", "4.0"))),
        "start_offset_seconds": float(payload.get("start_offset_seconds", os.getenv("FACE_SCAN_START_OFFSET_SECONDS", "0.0"))),
        "max_sampled_frames": int(payload.get("max_sampled_frames", os.getenv("FACE_SCAN_MAX_SAMPLED_FRAMES", "0"))),
        "min_clear_seconds": float(payload.get("min_clear_seconds", os.getenv("FACE_MIN_CLEAR_SECONDS", "2.0"))),
        "min_face_area_ratio": float(payload.get("min_face_area_ratio", os.getenv("FACE_MIN_AREA_RATIO", "0.06"))),
        "min_sharpness": float(payload.get("min_sharpness", os.getenv("FACE_MIN_SHARPNESS", "70.0"))),
        "min_brightness": float(payload.get("min_brightness", os.getenv("FACE_MIN_BRIGHTNESS", "40.0"))),
        "min_quality_score": float(payload.get("min_quality_score", os.getenv("FACE_MIN_QUALITY_SCORE", "0.55"))),
        "seed_acceptance_threshold": float(payload.get("seed_acceptance_threshold", os.getenv("FACE_SEED_ACCEPTANCE_THRESHOLD", "0.60"))),
        "max_aspect_ratio_deviation": float(payload.get("max_aspect_ratio_deviation", os.getenv("FACE_MAX_ASPECT_RATIO_DEVIATION", "0.65"))),
        "min_stability": float(payload.get("min_stability", os.getenv("FACE_MIN_STABILITY", "0.45"))),
        "dnn_confidence": float(
            payload.get(
                "dnn_confidence",
                os.getenv("FACE_DETECTOR_SCORE_THRESHOLD", os.getenv("FACE_DNN_CONFIDENCE", "0.65")),
            )
        ),
        "min_face_size_px": int(payload.get("min_face_size_px", os.getenv("FACE_MIN_SIZE_PX", "80"))),
        "verifier_enabled": _as_bool(payload.get("verifier_enabled"), _as_bool(os.getenv("FACE_VERIFIER_ENABLED"), True)),
        "verifier_score_threshold": float(payload.get("verifier_score_threshold", os.getenv("FACE_VERIFIER_SCORE_THRESHOLD", "0.92"))),
        "verifier_min_area_ratio": float(payload.get("verifier_min_area_ratio", os.getenv("FACE_VERIFIER_MIN_AREA_RATIO", "0.25"))),
        "verifier_max_center_offset": float(payload.get("verifier_max_center_offset", os.getenv("FACE_VERIFIER_MAX_CENTER_OFFSET", "0.45"))),
        "prefer_gpu": _as_bool(payload.get("prefer_gpu"), _as_bool(os.getenv("FACE_GPU_ENABLED"), True)),
        "gpu_device_id": int(payload.get("gpu_device_id", os.getenv("FACE_GPU_DEVICE_ID", "0"))),
        "detector_device": payload.get("detector_device", os.getenv("FACE_DETECTOR_DEVICE")),
        "verifier_device": payload.get("verifier_device", os.getenv("FACE_VERIFIER_DEVICE")),
        "embedding_device": payload.get("embedding_device", os.getenv("FACE_EMBEDDING_DEVICE")),
        "gpu_diagnostics": _as_bool(payload.get("gpu_diagnostics"), _as_bool(os.getenv("FACE_GPU_DIAGNOSTICS"), True)),
        "duplicate_similarity_threshold": float(payload.get("duplicate_similarity_threshold", os.getenv("FACE_SEED_DUPLICATE_SIMILARITY_THRESHOLD", "0.985"))),
        "write_debug_stats": _as_bool(payload.get("write_debug_stats"), _as_bool(os.getenv("FACE_SEED_DEBUG_STATS_ENABLED"), False)),
        "debug_stats_dir": payload.get("debug_stats_dir", os.getenv("FACE_SEED_DEBUG_STATS_DIR")),
    }

    t = threading.Thread(
        target=_run_scan_directory_bg,
        args=(str(root), production, skip_done, recursive, scan_kwargs),
        daemon=True,
        name=f"scan-dir-{root.name}",
    )
    t.start()
    return JSONResponse({"ok": True, "directory": str(root), "status": "scanning"})


@app.get("/api/probe/{video_id}")
def api_probe(video_id: int) -> JSONResponse:
    candidate = _resolve_video_id_path(video_id)
    return JSONResponse({"duration": _probe_duration(str(candidate))})


def _build_stream_cmd(
    input_path: str,
    seek: float,
    use_nvenc: bool,
) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if seek > 0:
        cmd += ["-ss", f"{seek:.3f}"]
    if use_nvenc:
        video_codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
    else:
        video_codec = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]
    cmd += [
        "-i", input_path,
        *video_codec,
        "-c:a", "aac",
        "-b:a", "128k",
        "-ac", "2",
        "-movflags", "frag_keyframe+empty_moov+default_base_moof",
        "-f", "mp4",
        "pipe:1",
    ]
    return cmd


@app.get("/stream/{video_id}")
async def stream_video(video_id: int, t: float = 0.0) -> StreamingResponse:
    candidate = _resolve_video_id_path(video_id)

    codecs_to_try: list[bool] = [True, False] if _NVENC_AVAILABLE else [False]
    proc: asyncio.subprocess.Process | None = None

    for use_nvenc in codecs_to_try:
        cmd = _build_stream_cmd(str(candidate), t, use_nvenc)
        encoder_name = "h264_nvenc" if use_nvenc else "libx264"
        try:
            candidate_proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail="FFmpeg not found on this server") from exc

        # Give ffmpeg up to 2 s to either start producing output or exit with an error.
        try:
            first_chunk = await asyncio.wait_for(candidate_proc.stdout.read(65536), timeout=2.0)  # type: ignore[union-attr]
        except asyncio.TimeoutError:
            first_chunk = b""

        if candidate_proc.returncode is not None and candidate_proc.returncode != 0:
            # Process already exited with error – try next codec.
            stderr_out = b""
            try:
                stderr_out = await asyncio.wait_for(candidate_proc.stderr.read(4096), timeout=1.0)  # type: ignore[union-attr]
            except asyncio.TimeoutError:
                pass
            logger.error(
                "ffmpeg exited immediately with %s (rc=%d): %s",
                encoder_name,
                candidate_proc.returncode,
                stderr_out.decode(errors="replace").strip(),
            )
            try:
                candidate_proc.kill()
            except ProcessLookupError:
                pass
            await candidate_proc.wait()
            continue

        if not first_chunk:
            # Process is running but produced no data in 2 s — encoder likely failed internally.
            stderr_out = b""
            try:
                stderr_out = await asyncio.wait_for(candidate_proc.stderr.read(4096), timeout=0.5)  # type: ignore[union-attr]
            except asyncio.TimeoutError:
                pass
            stderr_text = stderr_out.decode(errors="replace").strip()
            if stderr_text:
                logger.error("ffmpeg %s produced no data: %s", encoder_name, stderr_text)
            else:
                logger.error("ffmpeg %s produced no data after 2 s — skipping", encoder_name)
            try:
                candidate_proc.kill()
            except ProcessLookupError:
                pass
            await candidate_proc.wait()
            continue

        # Process is alive (or produced output) – use it.
        logger.info("Streaming %s with encoder=%s", candidate.name, encoder_name)
        proc = candidate_proc
        initial_data = first_chunk
        break

    if proc is None:
        raise HTTPException(status_code=502, detail="Stream startup failed for all available encoders")

    async def _iter():
        assert proc is not None
        assert proc.stdout is not None
        assert proc.stderr is not None
        try:
            if initial_data:
                yield initial_data
            while True:
                chunk = await proc.stdout.read(65536)
                if not chunk:
                    break
                yield chunk
        finally:
            # Drain stderr so the process can exit cleanly.
            try:
                stderr_tail = await asyncio.wait_for(proc.stderr.read(4096), timeout=1.0)
                if stderr_tail:
                    logger.warning("ffmpeg stderr tail: %s", stderr_tail.decode(errors="replace").strip())
            except asyncio.TimeoutError:
                pass
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
