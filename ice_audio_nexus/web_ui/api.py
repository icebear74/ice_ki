"""
ice_audio_nexus – web_ui/api.py

FastAPI-Backend für das interaktive Webinterface.

Funktionen:
  - Video-Streaming via FFmpeg (CUDA-beschleunigt, H.264/AAC für Browser)
  - Sprecher-Timeline aus der MariaDB
  - Echtzeit-Updates via WebSocket
  - REST-Endpunkte zum Benennen und Bestätigen von Sprechern
  - "Finalize Episode" – triggert Master-Vektor-Neuberechnung

Starten:
  python web_ui/api.py
  Browser: http://localhost:8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Projektpfade
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(
    title="ice_audio_nexus",
    description="KI-basierte Video-Audio-Analyse & Personenidentifikation",
    version="1.0.0",
)

# Templates & Static
_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# WebSocket-Verbindungen für Echtzeit-Updates
_ws_clients: list[WebSocket] = []


# ------------------------------------------------------------------
# Datenbank-Verbindung (lazy)
# ------------------------------------------------------------------

def _get_db():
    from db.database import get_connection, init_db
    return get_connection()


# ------------------------------------------------------------------
# Startvorgang: DB initialisieren
# ------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    try:
        from db.database import init_db
        init_db()
        logger.info("Datenbank initialisiert.")
    except Exception as e:
        logger.warning("DB-Initialisierung fehlgeschlagen (starte ohne DB): %s", e)


# ------------------------------------------------------------------
# Root – Webinterface
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ------------------------------------------------------------------
# Episoden-Liste
# ------------------------------------------------------------------

@app.get("/api/episodes")
async def list_episodes():
    """Gibt alle bekannten Episoden zurück."""
    try:
        conn = _get_db()
        from db.database import get_all_episodes
        episodes = get_all_episodes(conn)
        conn.close()
        return {"episodes": episodes}
    except Exception as e:
        logger.error("Episoden-Liste Fehler: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Sprecher-Timeline einer Episode
# ------------------------------------------------------------------

@app.get("/api/segments")
async def get_segments(
    series_name: str = Query(..., description="Serienname"),
    episode_title: str = Query(..., description="Episodentitel"),
):
    """
    Gibt alle Sprecher-Segmente einer Episode zurück.
    Die Zeitstempel (start_ms, end_ms) erlauben die Synchronisierung
    mit dem HTML5-Video-Player.
    """
    try:
        conn = _get_db()
        from db.database import get_segments_for_episode
        segments = get_segments_for_episode(conn, series_name, episode_title)
        conn.close()
        return {"segments": segments}
    except Exception as e:
        logger.error("Segmente Fehler: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Identitäten-Liste
# ------------------------------------------------------------------

@app.get("/api/identities")
async def list_identities():
    """Gibt alle bekannten Identitäten zurück."""
    try:
        conn = _get_db()
        from db.database import get_all_identities
        identities = get_all_identities(conn)
        conn.close()
        return {"identities": identities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Sprecher benennen / zuweisen (Rename / Assign)
# ------------------------------------------------------------------

class AssignRequest(BaseModel):
    series_name: str
    episode_title: str
    raw_speaker_id: str
    character_name: str
    series_context: str
    sync_actor_name: Optional[str] = None
    confirmed: bool = False


@app.post("/api/assign")
async def assign_speaker(req: AssignRequest):
    """
    Weist einer Sprecher-ID (raw_speaker_id) einen Charakternamen und
    Serien-Kontext zu. Erstellt bei Bedarf eine neue Identität in der DB.

    Aktualisiert alle Segmente der Episode mit dieser raw_speaker_id.
    """
    try:
        conn = _get_db()
        try:
            cur = conn.cursor()

            # Prüfen, ob Identität bereits existiert
            cur.execute(
                """
                SELECT i.id, i.voice_id FROM identities i
                WHERE i.character_name = %s AND i.series_name = %s
                """,
                (req.character_name, req.series_context),
            )
            row = cur.fetchone()

            if row:
                identity_id = row[0]
            else:
                # Neues voice_profile anlegen (Platzhalter-Vektor)
                from db.database import upsert_voice_profile
                placeholder = [0.0] * 512
                voice_id = upsert_voice_profile(conn, placeholder, sample_count=0)

                # Neue Identität anlegen
                cur.execute(
                    """
                    INSERT INTO identities
                        (voice_id, character_name, series_name, sync_actor_name)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (voice_id, req.character_name, req.series_context, req.sync_actor_name),
                )
                conn.commit()
                identity_id = cur.lastrowid

            # Alle Segmente der Episode mit dieser raw_speaker_id aktualisieren
            from db.database import assign_identity_to_speaker
            updated = assign_identity_to_speaker(
                conn,
                req.series_name,
                req.episode_title,
                req.raw_speaker_id,
                identity_id,
                confirmed=req.confirmed,
            )

            conn.close()

            # WebSocket-Broadcast: Live-Update an alle verbundenen Browser
            await _broadcast(
                {
                    "type": "speaker_assigned",
                    "raw_speaker_id": req.raw_speaker_id,
                    "identity_id": identity_id,
                    "character_name": req.character_name,
                    "series_context": req.series_context,
                    "updated_segments": updated,
                }
            )

            logger.info(
                "Sprecher '%s' → '%s' (%s) zugewiesen (%d Segmente)",
                req.raw_speaker_id,
                req.character_name,
                req.series_context,
                updated,
            )
            return {
                "success": True,
                "identity_id": identity_id,
                "updated_segments": updated,
            }
        except Exception as e:
            conn.close()
            raise e
    except Exception as e:
        logger.error("Assign-Fehler: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Confirm – Zuordnung bestätigen
# ------------------------------------------------------------------

class ConfirmRequest(BaseModel):
    series_name: str
    episode_title: str
    raw_speaker_id: str


@app.post("/api/confirm")
async def confirm_speaker(req: ConfirmRequest):
    """
    Bestätigt die Zuordnung einer Sprecher-ID. Setzt is_confirmed=TRUE
    für alle Segmente dieser raw_speaker_id in der Episode.
    """
    try:
        conn = _get_db()
        cur = conn.cursor()

        # Aktuelle Identität ermitteln
        cur.execute(
            """
            SELECT identity_id FROM episode_segments
            WHERE series_name = %s AND episode_title = %s AND raw_speaker_id = %s
            LIMIT 1
            """,
            (req.series_name, req.episode_title, req.raw_speaker_id),
        )
        row = cur.fetchone()
        if not row or not row[0]:
            conn.close()
            raise HTTPException(
                status_code=400,
                detail="Kein identity_id für diesen Sprecher gefunden. Erst zuweisen.",
            )

        identity_id = row[0]

        from db.database import assign_identity_to_speaker
        updated = assign_identity_to_speaker(
            conn,
            req.series_name,
            req.episode_title,
            req.raw_speaker_id,
            identity_id,
            confirmed=True,
        )
        conn.close()

        await _broadcast(
            {
                "type": "speaker_confirmed",
                "raw_speaker_id": req.raw_speaker_id,
                "identity_id": identity_id,
                "updated_segments": updated,
            }
        )

        return {"success": True, "updated_segments": updated}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Confirm-Fehler: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Finalize Episode – Master-Vektor neu berechnen
# ------------------------------------------------------------------

class FinalizeRequest(BaseModel):
    series_name: str
    episode_title: str


@app.post("/api/finalize")
async def finalize_episode(req: FinalizeRequest):
    """
    Triggert die Master-Vektor-Neuberechnung nach abgeschlossenem Labeling.
    Verbessert die Erkennungsgenauigkeit für zukünftige Episoden.
    """
    try:
        from processor.scanner import recompute_master_vectors
        recompute_master_vectors(req.series_name, req.episode_title)

        await _broadcast(
            {
                "type": "episode_finalized",
                "series_name": req.series_name,
                "episode_title": req.episode_title,
            }
        )

        return {"success": True, "message": "Master-Vektoren aktualisiert."}
    except Exception as e:
        logger.error("Finalize-Fehler: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Video-Streaming via FFmpeg (CUDA, H.264 → Browser-kompatibel)
# ------------------------------------------------------------------

@app.get("/api/stream")
async def stream_video(
    video_path: str = Query(..., description="Absoluter Pfad zur Videodatei"),
    seek: float = Query(0.0, description="Startzeit in Sekunden"),
):
    """
    Streamt das Video browsergerecht (H.264/AAC, fragmented MP4).
    Unterstützt Seekable-Streams via ?seek= Parameter.

    FFmpeg nutzt CUDA-Hardwareenkodierung (h264_nvenc) wenn verfügbar,
    sonst Fallback auf libx264.
    """
    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail="Videodatei nicht gefunden.")

    # Validate that the resolved path has a recognised video extension to prevent
    # path-injection attacks from serving arbitrary files.
    _ALLOWED_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".wmv", ".m4v", ".ts", ".webm"}
    if Path(video_path).suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Dateityp nicht unterstützt.")

    def _build_ffmpeg_cmd(use_cuda: bool) -> list[str]:
        encoder = "h264_nvenc" if use_cuda else "libx264"
        preset = "p4" if use_cuda else "veryfast"
        return [
            "ffmpeg",
            "-ss", str(seek),
            "-i", video_path,
            "-c:v", encoder,
            "-preset", preset,
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "frag_keyframe+empty_moov+faststart",
            "-f", "mp4",
            "pipe:1",
        ]

    async def video_generator():
        cmd = _build_ffmpeg_cmd(use_cuda=True)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            while True:
                chunk = await proc.stdout.read(65536)
                if not chunk:
                    break
                yield chunk
        except asyncio.CancelledError:
            proc.kill()
            raise
        finally:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()

    return StreamingResponse(
        video_generator(),
        media_type="video/mp4",
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


# ------------------------------------------------------------------
# WebSocket für Echtzeit-Updates
# ------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket-Verbindung für Echtzeit-Updates.
    Sendet JSON-Events bei Sprecher-Zuweisungen, Bestätigungen und
    Episode-Finalisierungen an alle verbundenen Browser.
    """
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("WebSocket verbunden (aktive Clients: %d)", len(_ws_clients))
    try:
        while True:
            # Hält die Verbindung offen; Client kann ping/pong senden
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)
        logger.info("WebSocket getrennt (aktive Clients: %d)", len(_ws_clients))


async def _broadcast(message: dict) -> None:
    """Sendet eine JSON-Nachricht an alle verbundenen WebSocket-Clients."""
    dead: list[WebSocket] = []
    for ws in list(_ws_clients):
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ------------------------------------------------------------------
# Entry-Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        app_dir=str(Path(__file__).parent),
    )
