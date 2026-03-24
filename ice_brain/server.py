"""
ice_brain – FastAPI server

Entry point:  python server.py
              uvicorn server:app --host 0.0.0.0 --port 8000

Startup sequence
----------------
1. Load GGUF models (router on P4, main on P100).
2. Initialise MySQL connection pool + run schema if needed.
3. Serve static WebUI from web/.
4. Accept OpenAI-compatible POST /v1/chat/completions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Ensure the ice_brain directory itself is on sys.path so sibling imports work
# when running as `python server.py` from any directory.
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import re
import secrets
from dataclasses import dataclass, field

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from db.connection import init_db
from llm_manager import LLMManager
from models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    LoginRequest,
    LoginResponse,
    SetPasswordRequest,
    UsageInfo,
)
from router import IntentRouter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_DIR = _HERE / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            _LOG_DIR / "ice_brain.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("ice_brain")

# ---------------------------------------------------------------------------
# App + session store
# ---------------------------------------------------------------------------
app = FastAPI(title="ice_brain", version="0.1.0")


# ---------------------------------------------------------------------------
# Session management mit Timeout
# ---------------------------------------------------------------------------

@dataclass
class _Session:
    user_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def is_expired(self, timeout_min: int = 30) -> bool:
        return (time.time() - self.last_active) > (timeout_min * 60)

    def touch(self) -> None:
        self.last_active = time.time()


# token → Session  (in-memory; cleared on restart)
_sessions: dict[str, _Session] = {}

# Globals – populated during startup
llm_manager: LLMManager = LLMManager()
intent_router: IntentRouter | None = None


def _new_token(user_id: str) -> str:
    token = secrets.token_hex(32)
    _sessions[token] = _Session(user_id=user_id)
    return token


def _resolve_token(token: str | None) -> str | None:
    """Return user_id for *token*, or None if invalid/missing/expired."""
    if not token:
        return None
    session = _sessions.get(token)
    if session is None:
        return None
    # Timeout aus config lesen (Standard: 30 Minuten)
    try:
        import config as _cfg  # noqa: PLC0415
        timeout = getattr(_cfg, "SESSION_TIMEOUT_MIN", 30)
    except ImportError:
        timeout = 30
    if session.is_expired(timeout):
        del _sessions[token]
        logger.debug("Session %s...%s für user %r abgelaufen und entfernt.", token[:8], token[-4:], session.user_id)
        return None
    session.touch()
    return session.user_id


def _cleanup_expired_sessions() -> None:
    """Abgelaufene Sessions aus dem Speicher entfernen."""
    try:
        import config as _cfg  # noqa: PLC0415
        timeout = getattr(_cfg, "SESSION_TIMEOUT_MIN", 30)
    except ImportError:
        timeout = 30
    expired = [t for t, s in _sessions.items() if s.is_expired(timeout)]
    for t in expired:
        del _sessions[t]
    if expired:
        logger.debug("Cleanup: %d abgelaufene Session(s) entfernt.", len(expired))


def _get_user_role(user_id: str) -> str:
    """Return the role of *user_id* from DB, default 'user'."""
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT role FROM users WHERE user_id = %s", (user_id,))
            row = cursor.fetchone()
            cursor.close()
        return row[0] if row else "user"
    except Exception:  # noqa: BLE001
        return "user"


# ---------------------------------------------------------------------------
# Brute-force protection
# ---------------------------------------------------------------------------

@dataclass
class _LoginAttemptTracker:
    """Tracks failed login attempts per IP address."""
    failed_count: int = 0
    first_attempt: float = field(default_factory=time.time)
    blocked_until: float = 0.0


_login_attempts: dict[str, _LoginAttemptTracker] = {}
_blocked_ips_log: list[dict] = []  # Global memory for admin notifications

_MAX_FAILED_ATTEMPTS = 5
_BLOCK_DURATION_SEC = 60 * 60  # 60 minutes


def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For behind a reverse proxy."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_ip_blocked(ip: str) -> bool:
    tracker = _login_attempts.get(ip)
    if tracker is None:
        return False
    if tracker.blocked_until > time.time():
        return True
    # Block expired – reset
    if tracker.blocked_until > 0:
        del _login_attempts[ip]
    return False


def _record_failed_login(ip: str, username: str) -> bool:
    """Record a failed attempt. Returns True if the IP is now blocked."""
    now = time.time()
    tracker = _login_attempts.get(ip)
    if tracker is None:
        tracker = _LoginAttemptTracker(failed_count=1, first_attempt=now)
        _login_attempts[ip] = tracker
        return False
    # Reset if older than block duration
    if now - tracker.first_attempt > _BLOCK_DURATION_SEC:
        tracker.failed_count = 1
        tracker.first_attempt = now
        return False
    tracker.failed_count += 1
    if tracker.failed_count >= _MAX_FAILED_ATTEMPTS:
        tracker.blocked_until = now + _BLOCK_DURATION_SEC
        _blocked_ips_log.append({
            "ip": ip,
            "blocked_at": datetime.now(timezone.utc).isoformat(),
            "attempts": tracker.failed_count,
            "last_username": username,
        })
        logger.warning(
            "IP %s blocked for 60 min after %d failed login attempts (last user: %r).",
            ip, tracker.failed_count, username,
        )
        return True
    return False


def _clear_failed_logins(ip: str) -> None:
    """Clear tracker on successful login."""
    _login_attempts.pop(ip, None)


def _pop_blocked_ips_for_admin() -> list[dict]:
    """Return and clear all blocked IP notifications for admin display."""
    if not _blocked_ips_log:
        return []
    items = list(_blocked_ips_log)
    _blocked_ips_log.clear()
    return items


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

def _reset_wiki_cache_if_no_images() -> None:
    """One-time migration: clear stale wiki cache so it is rebuilt with image_url.

    Runs only when wiki_cache contains rows but NONE have image_url populated —
    i.e. the first startup after the image pipeline was added.  Subsequent
    startups are no-ops because at least one row will have image_url set.
    """
    from db.connection import get_connection  # noqa: PLC0415
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM wiki_cache")
        total = cursor.fetchone()[0]
        if total == 0:
            cursor.close()
            logger.info("Wiki cache is empty – no reset needed.")
            return
        cursor.execute("SELECT COUNT(*) FROM wiki_cache WHERE image_url IS NOT NULL")
        with_image = cursor.fetchone()[0]
        cursor.close()

    if with_image > 0:
        logger.info(
            "Wiki cache already has %d/%d entries with image_url – skipping one-time reset.",
            with_image, total,
        )
        return

    logger.info(
        "Wiki cache has %d entries but none have image_url – clearing for rebuild.", total
    )
    try:
        from db.connection import get_connection as _gc  # noqa: PLC0415
        with _gc() as conn:
            cursor = conn.cursor()
            # wiki_chunks has no FK cascade, delete explicitly first
            cursor.execute("DELETE FROM wiki_chunks")
            cursor.execute("DELETE FROM wiki_cache")
            # Reset enrichment flags so the enrichment worker rebuilds the cache
            cursor.execute("UPDATE user_memory SET enriched = FALSE, enriched_at = NULL")
            cursor.execute("UPDATE relation_memory SET enriched = FALSE, enriched_at = NULL")
            conn.commit()
            cursor.close()
        logger.info("Wiki cache cleared and enrichment flags reset – rebuild will start automatically.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Wiki cache reset failed: %s", exc)


@app.on_event("startup")
async def startup() -> None:
    # 1. Load config
    try:
        import config  # noqa: PLC0415
        models_cfg = config.MODELS
    except ImportError:
        logger.error(
            "config.py not found!  Run:  cp config.py.example config.py  "
            "and fill in your model paths and MySQL credentials."
        )
        models_cfg = {}

    # Apply HF_TOKEN from config to environment (if set and not already in env)
    try:
        import config as _cfg_hf  # noqa: PLC0415
        hf_token = getattr(_cfg_hf, "HF_TOKEN", "")
        if hf_token and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = hf_token
            logger.info("HF_TOKEN set from config.py.")
    except ImportError:
        pass

    # Apply DEBUG_LOGGING flag: when True, set all ice_brain-related loggers to DEBUG level.
    try:
        import config as _cfg_debug  # noqa: PLC0415
        if getattr(_cfg_debug, "DEBUG_LOGGING", False):
            for _log_name in ("ice_brain", "router", "db.memory", "db.wiki", "workers.enrichment", "tools.wikipedia"):
                logging.getLogger(_log_name).setLevel(logging.DEBUG)
            # Also lower the root handler threshold so DEBUG messages are actually emitted.
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("DEBUG_LOGGING enabled – verbose logging active.")
    except ImportError:
        pass

    # 2. Load LLMs (failures are logged but don't abort startup)
    if models_cfg:
        llm_manager.load_all(models_cfg)
    else:
        logger.warning("No model config – server starts without LLMs.")

    # 3. Set up router
    global intent_router  # noqa: PLW0603
    intent_router = IntentRouter(llm_manager, model_name="router")

    # 4. Init DB (failures are also logged, not fatal for startup)
    try:
        init_db()
    except Exception as exc:  # noqa: BLE001
        logger.error("DB init failed: %s", exc)
        logger.error(
            "The server will run without database support until MySQL is reachable."
        )

    # 4b. One-time wiki cache rebuild: clear old entries that have no image_url so
    #     the enrichment worker re-fetches them with the new image pipeline.
    #     Only runs when wiki_cache rows exist but NONE have image_url populated yet.
    try:
        _reset_wiki_cache_if_no_images()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Wiki cache reset check failed (non-fatal): %s", exc)

    # 5. Ensure admin user exists
    try:
        from db.users import ensure_admin_user  # noqa: PLC0415
        admin_username = getattr(config, "ADMIN_USER", "admin") if "config" in sys.modules else "admin"
        ensure_admin_user(admin_username)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not ensure admin user: %s", exc)

    logger.info("ice_brain ready.  Model status: %s", llm_manager.get_status())

    # 6. Start background enrichment loop
    try:
        import asyncio  # noqa: PLC0415
        from workers.enrichment import enrichment_loop  # noqa: PLC0415
        asyncio.ensure_future(enrichment_loop(llm_manager))
        logger.info("Enrichment background loop registered.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not start enrichment loop: %s", exc)

    # 7. Start background cleanup loop
    try:
        from workers.cleanup import cleanup_loop  # noqa: PLC0415
        asyncio.ensure_future(cleanup_loop(llm_manager))
        logger.info("Cleanup background loop registered.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not start cleanup loop: %s", exc)

    # 7. Pre-load embedding model so download/disk errors surface at startup
    try:
        from tools.embeddings import configure_embedding_device, load_embedding_model  # noqa: PLC0415
        emb_cfg = getattr(config, "EMBEDDING_MODEL", {}) if "config" in sys.modules else {}
        emb_gpu = emb_cfg.get("gpu") if isinstance(emb_cfg, dict) else None
        if emb_gpu is not None:
            configure_embedding_device(f"cuda:{emb_gpu}")
            logger.info("Embedding model will use GPU %d (cuda:%d).", emb_gpu, emb_gpu)
        load_embedding_model()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedding model pre-load raised unexpected error: %s", exc)


# ---------------------------------------------------------------------------
# Background helpers
# ---------------------------------------------------------------------------

def _log_conversation_sync(user_id: str, user_msg: str, assistant_msg: str, intent: str) -> None:
    """Write conversation to DB (runs in background thread)."""
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversation_log (user_id, role, content, model_used, intent) "
                "VALUES (%s, %s, %s, %s, %s)",
                (user_id, "user", user_msg, "main", intent),
            )
            cursor.execute(
                "INSERT INTO conversation_log (user_id, role, content, model_used, intent) "
                "VALUES (%s, %s, %s, %s, %s)",
                (user_id, "assistant", assistant_msg, "main", intent),
            )
            conn.commit()
            cursor.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to log conversation: %s", exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "models": llm_manager.get_status()}


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.post("/auth/login", response_model=LoginResponse)
async def auth_login(req: LoginRequest, request: Request) -> LoginResponse:
    from db.users import authenticate  # noqa: PLC0415
    from db.connection import get_connection  # noqa: PLC0415

    client_ip = _get_client_ip(request)

    # Check if IP is blocked
    if _is_ip_blocked(client_ip):
        logger.warning("Blocked IP %s tried to login as %r.", client_ip, req.username)
        return JSONResponse(
            status_code=429,
            content={"error": "Zu viele fehlgeschlagene Anmeldeversuche. IP für 60 Minuten gesperrt."},
        )

    result = authenticate(req.username, req.password)
    if result is None:
        now_blocked = _record_failed_login(client_ip, req.username)
        if now_blocked:
            return JSONResponse(
                status_code=429,
                content={"error": "Zu viele fehlgeschlagene Anmeldeversuche. IP für 60 Minuten gesperrt."},
            )
        return JSONResponse(status_code=401, content={"error": "Ungültiger Benutzername oder Passwort."})

    # Success – clear attempts
    _clear_failed_logins(client_ip)

    user_id, first_login = result

    # Fetch role for response
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT role FROM users WHERE user_id = %s", (user_id,))
            row = cursor.fetchone()
            cursor.close()
        role = row[0] if row else "user"
    except Exception:  # noqa: BLE001
        role = "user"

    # Check for blocked IP notifications if user is admin
    blocked_notifications = None
    if role == "admin":
        alerts = _pop_blocked_ips_for_admin()
        if alerts:
            blocked_notifications = alerts

    token = None if first_login else _new_token(user_id)
    return LoginResponse(
        user_id=user_id,
        username=req.username,
        role=role,
        first_login=first_login,
        token=token,
        security_alerts=blocked_notifications,
    )


@app.post("/auth/set-password")
async def auth_set_password(req: SetPasswordRequest) -> dict:
    from db.users import is_first_login, set_password  # noqa: PLC0415
    from db.connection import get_connection  # noqa: PLC0415

    if not req.new_password or len(req.new_password) < 8:
        return JSONResponse(status_code=400, content={"error": "Passwort muss mindestens 8 Zeichen haben."})

    try:
        if not is_first_login(req.user_id):
            return JSONResponse(status_code=403, content={"error": "Passwort bereits gesetzt."})
        set_password(req.user_id, req.new_password)
    except Exception as exc:  # noqa: BLE001
        logger.error("set-password error: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT username, role FROM users WHERE user_id = %s", (req.user_id,))
            row = cursor.fetchone()
            cursor.close()
        username = row[0] if row else req.user_id
        role = row[1] if row else "user"
    except Exception:  # noqa: BLE001
        username, role = req.user_id, "user"

    token = _new_token(req.user_id)
    return {"ok": True, "user_id": req.user_id, "username": username, "role": role, "token": token}


_WIKI_SNIPPET_MAX_CHARS = 800  # max characters per wiki chunk injected into the system prompt
_MAX_TOPIC_WORDS = 4           # max words to keep when extracting a search topic
_SHORT_MSG_WORD_THRESHOLD = 8  # messages with ≤ this many words are treated as follow-ups

# ---------------------------------------------------------------------------
# Streaming thinking-block filter
# ---------------------------------------------------------------------------

class _StreamThinkingFilter:
    """Stateful filter that strips <think>…</think> blocks from a token stream.

    Chunks arrive one at a time via :meth:`feed`.  A partial tag that straddles
    two chunks is handled by keeping a small look-ahead buffer.  Call
    :meth:`flush` once the stream is done to emit any buffered remainder.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self, strip: bool = True) -> None:
        self._strip = strip
        self._buf = ""
        self._in_think = False

    def feed(self, text: str) -> str:
        if not self._strip:
            return text
        self._buf += text
        out: list[str] = []
        while True:
            if self._in_think:
                end = self._buf.find(self._CLOSE)
                if end == -1:
                    # Still inside a think block – discard everything buffered.
                    self._buf = ""
                    break
                # Found closing tag – discard up to and including </think>.
                self._buf = self._buf[end + len(self._CLOSE):]
                self._in_think = False
            else:
                start = self._buf.find(self._OPEN)
                if start == -1:
                    # No opening tag – but keep the last (len(_OPEN)-1) chars
                    # buffered in case an opening tag straddles two chunks.
                    safe = max(0, len(self._buf) - (len(self._OPEN) - 1))
                    out.append(self._buf[:safe])
                    self._buf = self._buf[safe:]
                    break
                # Emit everything before <think>, then enter think-block mode.
                out.append(self._buf[:start])
                self._buf = self._buf[start + len(self._OPEN):]
                self._in_think = True
        return "".join(out)

    def flush(self) -> str:
        if not self._strip:
            remaining = self._buf
            self._buf = ""
            return remaining
        if self._in_think:
            # Incomplete think block at end of stream – discard.
            self._buf = ""
            self._in_think = False
            return ""
        remaining = self._buf
        self._buf = ""
        return remaining


# ---------------------------------------------------------------------------
# Correction detection + live Wikipedia lookup
# ---------------------------------------------------------------------------

# Anti-Halluzinations-Hinweis für den System-Prompt
_ANTI_HALLUCINATION_NOTE = (
    "\n\n⚠️ WICHTIGE REGELN:"
    "\n- Du hast KEINEN Internetzugang und kannst KEINE Websuchen durchführen."
    "\n- Erfinde NIEMALS URLs, Links, Webseiten oder Suchergebnisse."
    "\n- Wenn du etwas nicht weißt, sage es ehrlich. Halluziniere keine Fakten."
    "\n- Nutze [WIKI_SEARCH: ...] wenn du Fakten nachschlagen musst."
    "\n- Wenn du Wikipedia-Daten erhältst, nutze NUR diese als Faktenquelle."
    "\n- Nutze [WEB_SEARCH: ...] oder [NEWS_SEARCH: ...] für aktuelle Infos, Nachrichten und Sport."
    "\n- Wenn du Informationen aus Tools (Web, Wiki) verwendest, zitiere die Quelle immer"
    " mit einem Markdown-Link, z.B. [Artikelname](https://...) oder [Quelle](https://...)."
    "\n- Gib KEINE erfundenen Quellenangaben oder Links an."
    "\n- Antworte IMMER in der Sprache, in der der Benutzer schreibt."
    "\n  Wenn er deutsch schreibt, antworte auf deutsch."
    "\n  Wenn er englisch schreibt, antworte auf englisch. Passe dich dynamisch an."
)

# Phrases that signal the user is correcting the AI (German + common English)
_CORRECTION_RE = re.compile(
    r"(?:"
    r"da\s+bist\s+du\s+(?:aber\s+)?falsch"
    r"|du\s+liegst\s+(?:da\s+)?falsch"
    r"|das\s+stimmt\s+(?:so\s+)?nicht"
    r"|das\s+ist\s+(?:nicht|falsch|inkorrekt|wrong)"
    r"|du\s+(?:hast|liegst|bist)[^.!?]{0,40}(?:falsch|unrecht|irr(?:st|tu))"
    r"|falsch\s+informiert"
    r"|nicht\s+(?:korrekt|richtig|stimmt)"
    r"|aktualisier(?:e)?\s+dich"
    r"|inform(?:iere?)?\s+dich\s+(?:mal|bitte|doch)?"
    r"|schlag\s+(?:das\s+)?(?:mal\s+)?nach"
    r"|check\s+(?:das\s+)?(?:mal\s+)?nach"
    r"|wiki\s+(?:abfragen|nachschauen|nachschlagen)"
    r"|update\s+(?:dich|dein\s+wissen)"
    r"|korrigier(?:e)?\s+(?:dich|deine\s+(?:info|infos|angaben?))"
    # Death / obituary corrections
    r"|(?:er|sie|es)\s+ist\s+(?:gestorben|tot|verstorben)"
    r"|ist\s+(?:doch\s+)?(?:schon\s+)?(?:gestorben|tot|verstorben)"
    r"|(?:er|sie)\s+starb\b"
    r"|ist\s+(?:letztes?\s+(?:jahr|monat)|letzte\s+woche|kürzlich|neulich|gerade)\s+(?:gestorben|verstorben)"
    # "steht im Wiki / Wikipedia"
    r"|steht\s+(?:doch\s+)?(?:so\s+)?(?:auch\s+)?im\s+(?:wiki|wikipedia)"
    r"|steht\s+(?:auch\s+)?(?:so\s+)?(?:doch\s+)?im\s+(?:wiki|wikipedia)"
    r"|(?:das\s+)?steht\s+im\s+(?:artikel|wiki|wikipedia)"
    # "hast du vergessen"
    r"|hast\s+du\s+(?:das\s+)?vergessen"
    r"|das\s+hast\s+du\s+vergessen"
    r"|du\s+hast\s+(?:das\s+)?(?:wichtigste\s+)?vergessen"
    r")",
    re.IGNORECASE,
)

# Stopwords to strip when extracting a search topic from the correction message
_STOPWORDS = frozenset({
    # Articles
    "ein", "eine", "einer", "einen", "einem", "eines",
    "der", "die", "das", "dem", "den", "des",
    # Pronouns
    "du", "ich", "er", "sie", "es", "wir", "ihr", "man",
    "mich", "mir", "sich", "uns", "euch",
    "dich", "dein", "deine", "deiner", "deinem", "deinen",
    "mein", "meine", "meinem", "meinen", "meiner",
    # Common verbs
    "ist", "sind", "war", "waren", "wird", "werden", "wurde", "wurden",
    "hat", "haben", "hatte", "hatten", "habe", "hast",
    "sein", "bist", "bin", "seid",
    "liegst", "stehen", "steht", "standen", "kommen", "gehen", "machen",
    "sagen", "denk", "denke", "denken", "glaub", "glaube", "glauben",
    "weiß", "weißt", "wissen", "kann", "kannst", "können", "konnten",
    "musst", "muss", "müssen", "soll", "sollst", "sollen",
    "darf", "darfst", "dürfen", "magst", "möchte", "möchtest",
    "stimmt", "stimmen", "stimmst",
    # Prepositions / conjunctions
    "über", "unter", "nach", "vor", "mit", "ohne", "von", "beim",
    "aus", "bei", "für", "an", "auf", "in", "zu", "zum", "zur",
    "durch", "gegen", "zwischen", "neben", "hinter", "ab", "seit",
    "bis", "außer", "wegen", "trotz",
    "und", "oder", "aber", "doch", "mal", "bitte", "denn", "weil",
    "dass", "wann", "wenn", "ob", "wie", "was", "wer", "wo", "woher",
    # Adjectives / adverbs frequently appearing in corrections
    "nicht", "kein", "keine", "keiner", "nein", "ja",
    "falsch", "falsche", "falschen", "falscher", "falsches",
    "richtig", "richtige", "richtigen", "korrekt", "inkorrekt",
    "wrong", "incorrect", "false",
    "da", "dort", "hier", "hin", "her", "also", "noch", "schon",
    "eben", "doch", "halt", "nur", "sehr", "ganz", "viel", "alle",
    # Generic nouns that are not useful as search topics
    "informationen", "infos", "info", "angaben", "daten", "aussage",
    "fakt", "fakten", "sache", "sachen", "zeug",
    # Correction-related words
    "aktualisier", "informier", "schlag", "nach", "check",
    "wiki", "update", "korrigier",
})

# Phrases that signal the user wants a refresh/update of information
_UPDATE_RE = re.compile(
    r"(?:"
    r"aktualisier"
    r"|update"
    r"|refresh"
    r"|neu\s+laden"
    r"|lad(?:e)?\s+neu"
    r"|frisch(?:e|es)?\s+(?:daten|infos|wissen)"
    r"|hol\s+(?:dir\s+)?(?:neue|aktuelle)"
    r")",
    re.IGNORECASE,
)

# Einfache Begrüßungen / Small-Talk – keine Wiki-Suche nötig
_GREETING_RE = re.compile(
    r"^\s*(?:hallo|hi|hey|moin|guten\s+(?:morgen|tag|abend)|gute\s+nacht|"
    r"servus|grüß\s+(?:gott|dich)|tschüss?|bye|ciao|danke|bitte|ja|nein|ok(?:ay)?|"
    r"good\s+(?:morning|evening|night)|hello|thanks?|yes|no)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def _detect_correction(message: str) -> bool:
    """Return True when the user's message signals a factual correction."""
    return bool(_CORRECTION_RE.search(message))


def _extract_correction_topic(message: str) -> str:
    """Extract a short search topic from a correction message.

    Strips the correction phrases and stopwords, then returns the most
    meaningful remaining words as a search query (max 4 words).

    Strategy: prefer words that appear capitalised in the middle of a sentence
    (German nouns / proper nouns) over lowercase common words.
    """
    # Remove the correction trigger phrase
    cleaned = _CORRECTION_RE.sub(" ", message)
    # Tokenise (keep alphanumerics + umlauts) together with their original form
    raw_tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", cleaned)
    # Filter stopwords and very short tokens
    meaningful = [t for t in raw_tokens if t.lower() not in _STOPWORDS and len(t) > 2]
    if not meaningful:
        # Fallback: try whole message without correction phrases
        raw_tokens_full = re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", message)
        meaningful = [t for t in raw_tokens_full if t.lower() not in _STOPWORDS and len(t) > 2]
    if not meaningful:
        return message.strip()

    # Separate into capitalized (likely German nouns / proper nouns) and the rest.
    # Capitalised words in German tend to be the actual topics (nouns).
    caps = [t for t in meaningful if t[0].isupper()]
    lower = [t for t in meaningful if not t[0].isupper()]

    # Build query: capitalized tokens first (sorted by length desc), then others
    caps.sort(key=len, reverse=True)
    lower.sort(key=len, reverse=True)
    combined = caps + lower
    return " ".join(combined[:_MAX_TOPIC_WORDS])


def _extract_topic(message: str) -> str:
    """Extract a short search topic from any user message (not just corrections).

    Used for proactive wiki lookups when no cached chunks are found.
    Returns up to 4 meaningful words, preferring capitalised German nouns.
    """
    raw_tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", message)
    meaningful = [t for t in raw_tokens if t.lower() not in _STOPWORDS and len(t) > 2]
    if not meaningful:
        return ""
    caps = [t for t in meaningful if t[0].isupper()]
    lower = [t for t in meaningful if not t[0].isupper()]
    caps.sort(key=len, reverse=True)
    lower.sort(key=len, reverse=True)
    combined = caps + lower
    return " ".join(combined[:4])


# Patterns that suggest the user is asking about a specific topic
_TOPIC_QUESTION_RE = re.compile(
    r"(?:"
    r"kennst\s+du"
    r"|weiß(?:t)?\s+du"
    r"|sag\s+(?:mir|mal)"
    r"|erzähl"
    r"|was\s+(?:ist|sind|war|waren|bedeutet|weiß|macht|kann)"
    r"|wer\s+(?:ist|war|sind|waren)"
    r"|wo\s+(?:ist|liegt|befindet|gibt|war)"
    r"|wie\s+(?:ist|war|funktioniert|geht|heißt|alt)"
    r"|wann\s+(?:ist|war|wurde|hat|starb|geboren)"
    r"|can\s+you\s+tell"
    r"|do\s+you\s+know"
    r"|what\s+(?:is|are|was|were|does|did)"
    r"|who\s+(?:is|was|are)"
    r"|when\s+(?:is|was|did)"
    r"|where\s+(?:is|was|are)"
    r"|gibt\s+es"
    r"|handelt\s+(?:es\s+)?sich"
    r"|erkl(?:är|aer)"
    r"|beschreib"
    r"|nenne\s+(?:mir)?"
    r"|zeig\s+(?:mir)?"
    r"|such\s+(?:mir|mal)?"
    r"|info(?:rmation(?:en)?)?\s+(?:über|zu|von)"
    r"|\?\s*$"
    r")",
    re.IGNORECASE,
)

# Present-tense "who is" questions about current role holders — ALWAYS trigger a
# live Wikipedia lookup regardless of whether cached data already exists, because
# the cached article might have been written before the current person took office.
_LIVE_ALWAYS_RE = re.compile(
    r"(?:"
    # German present tense: "wer ist [der/die] [role]"
    r"wer\s+(?:ist|sind)\b"
    r"|wer\s+(?:war\s+)?(?:der|die)\s+(?:aktuelle?|derzeitige?|jetzige?)\b"
    r"|(?:der|die)\s+(?:aktuelle?|derzeitige?|jetzige?|neue?)\s+\w+\s+(?:ist|heißt|war|lautet)\b"
    # English present tense
    r"|who\s+is\b"
    r"|who\s+are\b"
    r"|who\s+(?:is|are)\s+(?:the\s+)?(?:current|new|latest)\b"
    r")",
    re.IGNORECASE,
)


def _live_wiki_context_for_correction(message: str, limit: int = 2, prior_context: str = "") -> str:
    """Fetch fresh Wikipedia data for the topic being corrected.

    Returns a formatted string to be injected at the top of the system prompt
    as a hard instruction the model MUST follow.  Returns an empty string when
    the lookup fails or yields nothing.

    When the correction message itself contains no recognisable topic (e.g.
    "Er ist gestorben"), *prior_context* (the last few user messages joined as
    a string) is used as a fallback to find the subject of the correction.
    """
    topic = _extract_correction_topic(message)
    if not topic and prior_context:
        topic = _extract_topic(prior_context)
    if not topic:
        return ""
    logger.info("Live wiki lookup triggered by correction. Query: %r", topic)
    try:
        from tools.wikipedia import wiki_live_lookup  # noqa: PLC0415
        results = wiki_live_lookup(topic, limit=limit)
        if not results:
            logger.info("Live wiki lookup: no results for %r.", topic)
            return ""
        lines = [
            "⚠️ FAKTENKORREKTUR – WIKIPEDIA LIVE (HÖCHSTE PRIORITÄT):",
            "Du MUSST die folgenden Fakten als korrekt akzeptieren und in deiner Antwort verwenden.",
            "Diese Daten wurden soeben live von Wikipedia abgerufen.",
            "Sie überschreiben dein Trainingswissen vollständig.",
            "Antworte AUSSCHLIESSLICH auf Basis dieser Daten – ignoriere jede gegenteilige Annahme aus deinem Training.",
            "",
        ]
        for r in results:
            snippet = (r.get("full_text") or r.get("summary", ""))[:1200].replace("\n", " ").strip()
            lines.append(f"[{r['title']}] {snippet}")
            if r.get("source_url"):
                title = r.get("title", "Wikipedia")
                lines.append(f"  Quelle: [{title} – Wikipedia]({r['source_url']})")
            if r.get("image_url"):
                img_src = _img_src_for_result(r)
                if img_src:
                    lines.append(f"  Bild: ![{r.get('title', '')}]({img_src})")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Live wiki lookup failed (non-fatal): %s", exc)
        return ""


def _live_wiki_context_proactive(message: str, limit: int = 2) -> str:
    """Fetch Wikipedia data for the topic of a message when no cached data exists.

    Called as a fallback when the vector search returns nothing relevant.
    Uses a softer label than the correction variant so the model knows the
    information is background context rather than a verified correction.
    Returns an empty string when the lookup fails or yields nothing.
    """
    topic = _extract_topic(message)
    if not topic:
        return ""
    logger.info("Proactive live wiki lookup. Query: %r", topic)
    try:
        from tools.wikipedia import wiki_live_lookup  # noqa: PLC0415
        # Pass the full original message to wiki_live_lookup so that person
        # follow-up logic can detect role keywords (Bürgermeister etc.).
        results = wiki_live_lookup(message, limit=limit)
        if not results:
            logger.info("Proactive live wiki lookup: no results for %r.", topic)
            return ""
        lines = [
            "📡 WIKIPEDIA-HINTERGRUNDWISSEN (live abgerufen, da kein lokaler Cache vorhanden):"
        ]
        for r in results:
            snippet = (r.get("full_text") or r.get("summary", ""))[:1000].replace("\n", " ").strip()
            lines.append(f"[{r['title']}] {snippet}")
            if r.get("source_url"):
                title = r.get("title", "Wikipedia")
                lines.append(f"  Quelle: [{title} – Wikipedia]({r['source_url']})")
            if r.get("image_url"):
                img_src = _img_src_for_result(r)
                if img_src:
                    lines.append(f"  Bild: ![{r.get('title', '')}]({img_src})")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Proactive live wiki lookup failed (non-fatal): %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Tool-use pattern parser (text-pattern based, no OpenAI function calling)
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r"\[(SEARCH_MEMORY|SEARCH_RELATION|WIKI_SEARCH|WEATHER|WEB_SEARCH|NEWS_SEARCH)\s*:\s*([^\]]{1,256})\]",
    re.IGNORECASE,
)

_IMG_MARKDOWN_RE = re.compile(r'!\[[^\]]*\]\((?:/api/image/|https?://)[^)]+\)')


def _img_src_for_result(r: dict) -> str | None:
    """Return the best image src for a wiki result dict.

    Tries to fetch-and-cache the image locally (returns /api/image/{id}).
    Falls back to the original Wikipedia URL when local caching fails so the
    image always renders for the user.  Returns None when no image is available.
    """
    image_url = r.get("image_url")
    if not image_url:
        return None
    try:
        from db.images import fetch_and_cache_url, link_image  # noqa: PLC0415
        img_id = fetch_and_cache_url(
            image_url, "wikipedia", r.get("title", "unknown"),
            alt_text=r.get("title", ""),
        )
        if img_id is not None:
            cache_id = r.get("id") or r.get("article_id")
            if cache_id is not None:
                link_image(img_id, "wiki_cache", cache_id)
            return f"/api/image/{img_id}?thumb=true"
    except Exception as exc_img:  # noqa: BLE001
        logger.warning("Wiki image caching failed (non-fatal): %s", exc_img)
    # Fallback: use original Wikipedia URL directly
    return image_url


def _extract_pending_images(*sections: str) -> list[str]:
    """Extract all /api/image/ Markdown snippets from context sections (deduplicated)."""
    seen: set[str] = set()
    images: list[str] = []
    for s in sections:
        for m in _IMG_MARKDOWN_RE.finditer(s):
            img = m.group()
            if img not in seen:
                seen.add(img)
                images.append(img)
    return images


def _parse_tool_calls(text: str) -> list[tuple[str, str]]:
    """Return list of (tool_name, query) tuples found in *text*."""
    return [(m.group(1).upper(), m.group(2).strip()) for m in _TOOL_CALL_RE.finditer(text)]


def _execute_tool_calls(
    tool_calls: list[tuple[str, str]],
    user_id: str,
) -> str:
    """Execute tool calls and return combined results as a formatted string."""
    if not tool_calls:
        return ""
    parts: list[str] = []
    for tool_name, query in tool_calls:
        try:
            if tool_name == "SEARCH_MEMORY":
                from db.memory import semantic_recall  # noqa: PLC0415
                results = semantic_recall(user_id, query, limit=5)
                if results:
                    lines = [f"[SEARCH_MEMORY: {query}]"]
                    for r in results:
                        lines.append(f"  - {r['content']} [{r.get('category','')}]")
                    parts.append("\n".join(lines))
            elif tool_name == "SEARCH_RELATION":
                from db.relations import find_relation, get_relation, get_relation_facts  # noqa: PLC0415
                relation_id = find_relation(user_id, query)
                if relation_id is not None:
                    rel = get_relation(relation_id)
                    facts = get_relation_facts(relation_id)
                    lines = [f"[SEARCH_RELATION: {query}]"]
                    if rel:
                        lines.append(f"  Name: {rel['name']}, Typ: {rel['relation_type']}")
                    for f in facts:
                        lines.append(f"  - {f['content']} [{f['category']}]")
                    parts.append("\n".join(lines))
            elif tool_name == "WIKI_SEARCH":
                from tools.wikipedia import wiki_live_lookup  # noqa: PLC0415
                results = wiki_live_lookup(query, limit=2)
                if results:
                    lines = [f"[WIKI_SEARCH: {query}]"]
                    for r in results:
                        snippet = (r.get("full_text") or r.get("summary", ""))[:600].replace("\n", " ")
                        lines.append(f"  [{r['title']}] {snippet}")
                        if r.get("source_url"):
                            title = r.get("title", "Wikipedia")
                            lines.append(f"  Quelle: [{title} – Wikipedia]({r['source_url']})")
                        if r.get("image_url"):
                            img_src = _img_src_for_result(r)
                            if img_src:
                                lines.append(f"  Bild: ![{r.get('title', '')}]({img_src})")
                    parts.append("\n".join(lines))
            elif tool_name == "WEATHER":
                from tools.weather import get_weather_for_user  # noqa: PLC0415
                result = get_weather_for_user(user_id, location_name=query if query else None)
                if result:
                    parts.append(f"[WEATHER: {query}]\n{result}")
            elif tool_name in ("WEB_SEARCH", "NEWS_SEARCH"):
                from tools.websearch import news_search, web_search  # noqa: PLC0415
                if tool_name == "NEWS_SEARCH":
                    results = news_search(query, max_results=5, timelimit="w")
                else:
                    results = web_search(query, max_results=5)
                if results:
                    lines = [f"[{tool_name}: {query}]"]
                    for r in results:
                        title = r.get("title", "")
                        url = r.get("url", "")
                        snippet = r.get("snippet", "")
                        date = r.get("date", "")
                        source = r.get("source", "")
                        entry = f"  [{title}]({url}) – {snippet}"
                        if date or source:
                            entry += f" ({source}, {date})" if source and date else f" ({source or date})"
                        lines.append(entry)
                    parts.append("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool call %s(%r) failed: %s", tool_name, query, exc)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# GPU stats helpers
# ---------------------------------------------------------------------------

_gpu_stats_cache: dict = {"data": None, "ts": 0.0}
_GPU_STATS_TTL = 2.0  # seconds between real queries


def _query_gpu_stats() -> dict:
    """Query per-GPU utilisation via pynvml (preferred) or nvidia-smi."""
    try:
        import pynvml  # noqa: PLC0415
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                temp: int | None = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:  # noqa: BLE001
                temp = None
            gpus.append({
                "index": i,
                "name": name,
                "util_pct": util.gpu,
                "mem_util_pct": util.memory,
                "mem_used_mb": mem.used // (1024 * 1024),
                "mem_total_mb": mem.total // (1024 * 1024),
                "temp_c": temp,
            })
        return {"gpus": gpus, "source": "nvml"}
    except Exception:  # noqa: BLE001
        pass

    # Fallback: nvidia-smi subprocess
    import subprocess  # noqa: PLC0415
    try:
        result = subprocess.run(  # noqa: S603
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5, check=False,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append({
                "index": int(parts[0]) if parts[0].isdigit() else 0,
                "name": parts[1],
                "util_pct": int(parts[2]) if parts[2].isdigit() else 0,
                "mem_util_pct": None,
                "mem_used_mb": int(parts[3]) if parts[3].isdigit() else 0,
                "mem_total_mb": int(parts[4]) if parts[4].isdigit() else 0,
                "temp_c": int(parts[5]) if parts[5].isdigit() else None,
            })
        return {"gpus": gpus, "source": "nvidia-smi"}
    except Exception as exc:  # noqa: BLE001
        return {"gpus": [], "error": str(exc)}





def _wiki_context_for_message(message: str, limit: int = 3, min_score: float = 0.55) -> str:
    """Search cached wiki chunks for *message* and format as a prompt section.

    Returns an empty string when no relevant chunks are found, when the
    embedding model is not yet loaded, or on any error (non-fatal).
    """
    if not message or len(message.strip()) < 4:
        return ""

    # Begrüßungen und Small-Talk überspringen – keine Wiki-Suche nötig
    if _GREETING_RE.match(message):
        return ""

    try:
        from db.wiki import search_wiki_chunks  # noqa: PLC0415
        logger.debug("Wiki search: querying chunks for message %r (limit=%d, min_score=%.2f)", message, limit, min_score)
        results = search_wiki_chunks(message, limit=limit)
        if not results:
            logger.debug("Wiki search: no chunks in DB (nothing indexed yet).")
            return ""
        logger.debug(
            "Wiki search: %d chunk(s) returned. Scores: %s",
            len(results),
            ", ".join(f"{r['title']!r}={r['score']:.3f}" for r in results),
        )
        relevant = [r for r in results if r["score"] >= min_score]
        if not relevant:
            logger.debug(
                "Wiki search: no chunks above threshold %.2f – best score was %.3f (%r).",
                min_score,
                results[0]["score"] if results else 0.0,
                results[0]["title"] if results else "",
            )
            return ""

        # Relevanzfilter: Prüfen ob signifikante Wörter der Nachricht in Chunk-Titeln vorkommen
        sig_words = [
            w.lower() for w in re.findall(r"[A-Za-zÄÖÜäöüß]{4,}", message)
            if w.lower() not in _STOPWORDS
        ]
        if sig_words:
            titles_lower = " ".join(r["title"].lower() for r in relevant)
            if not any(w in titles_lower for w in sig_words):
                logger.debug(
                    "Wiki search: Relevanztreffer-Titel passen nicht zur Anfrage (Wörter: %s) – verworfen.",
                    ", ".join(sig_words[:5]),
                )
                return ""

        logger.debug(
            "Wiki search: %d relevant chunk(s) injected into prompt: %s",
            len(relevant),
            ", ".join(f"{r['title']!r} (score={r['score']:.3f})" for r in relevant),
        )
        lines = [
            "📚 Relevantes Wikipedia-Hintergrundwissen "
            "(nutze es in deiner Antwort wenn passend, aber nur wenn es wirklich hilft):"
        ]
        for r in relevant:
            snippet = r["content"][:_WIKI_SNIPPET_MAX_CHARS].replace("\n", " ").strip()
            lines.append(f"[{r['title']}] {snippet}")

        # Inject cached images for each unique article (deduplicated by article_id)
        seen_article_ids: set[int] = set()
        for r in relevant:
            article_id = r.get("article_id")
            if not r.get("image_url") or not article_id or article_id in seen_article_ids:
                continue
            seen_article_ids.add(article_id)
            img_src = _img_src_for_result(r)
            if img_src:
                lines.append(f"  Bild: ![{r['title']}]({img_src})")

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Wiki context search failed (non-fatal): %s", exc)
        return ""



def _web_search_context(query: str, is_news: bool = False) -> str:
    """Perform a DuckDuckGo search and format the results as a prompt section.

    Returns a formatted string to be injected into the system prompt, or ""
    on error / when the package is not installed.
    """
    try:
        from tools.websearch import news_search, web_search  # noqa: PLC0415
        if is_news:
            results = news_search(query, max_results=5, timelimit="w")
            header = "📰 AKTUELLE NACHRICHTEN (DuckDuckGo, live abgerufen):"
        else:
            results = web_search(query, max_results=5)
            header = "🌐 WEB-SUCHERGEBNISSE (DuckDuckGo, live abgerufen):"
        if not results:
            logger.info("Web search returned no results for query %r.", query)
            return ""
        lines = [
            header,
            "Nutze diese Ergebnisse als Grundlage deiner Antwort und zitiere die Quellen mit Markdown-Links.",
            "",
        ]
        for r in results:
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            date = r.get("date", "")
            source = r.get("source", "")
            meta = f" ({source}, {date})" if source and date else (f" ({source})" if source else (f" ({date})" if date else ""))
            if url:
                lines.append(f"- **[{title}]({url})**{meta}: {snippet}")
            else:
                lines.append(f"- **{title}**{meta}: {snippet}")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Web search context failed (non-fatal): %s", exc)
        return ""


@app.get("/v1/gpu-stats")
async def get_gpu_stats() -> dict:
    """Return per-GPU utilisation stats (cached for up to 2 s)."""
    now = time.monotonic()
    if _gpu_stats_cache["data"] is not None and now - _gpu_stats_cache["ts"] < _GPU_STATS_TTL:
        return _gpu_stats_cache["data"]  # type: ignore[return-value]
    data = await asyncio.get_running_loop().run_in_executor(None, _query_gpu_stats)
    _gpu_stats_cache["data"] = data
    _gpu_stats_cache["ts"] = now
    return data


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
) -> ChatCompletionResponse:
    # Abgelaufene Sessions periodisch bereinigen
    _cleanup_expired_sessions()

    # Resolve authenticated user from session token (preferred) or fallback field.
    authed_user_id = _resolve_token(request.session_token)
    user_id = authed_user_id or request.user or "default"
    last_message = request.messages[-1].content if request.messages else ""

    # ── Help command: "help" / "hilfe" ────────────────────────────────────
    if re.match(r"^\s*(help|hilfe)\s*$", last_message, re.IGNORECASE):
        is_admin = bool(authed_user_id and _get_user_role(authed_user_id) == "admin")
        help_lines = [
            "**ice_brain – Verfügbare Befehle**",
            "",
            "**Allgemeine Befehle** (für alle Benutzer):",
            "  `help` / `hilfe` – Diese Hilfeseite anzeigen",
            "",
            "**Konversation:**",
            "  Schreibe einfach deine Frage oder Nachricht – ice_brain antwortet.",
            "  Persönliche Informationen werden automatisch gespeichert und später abgerufen.",
        ]
        if is_admin:
            help_lines += [
                "",
                "**Administrator-Befehle:**",
                "  `lege benutzer an: <Name>` – Neuen Benutzer anlegen",
                "    Der neue Benutzer kann sich sofort einloggen; beim ersten Login wird ein Passwort gesetzt.",
                "    Beispiel: `lege benutzer an: Maria Müller`",
                "  `anreicherung starten` – Gespeicherte Erinnerungen sofort mit Wikipedia-Wissen anreichern",
            ]
        else:
            help_lines += [
                "",
                "*(Administrator-Befehle sind nur für Admins sichtbar.)*",
            ]
        reply = "\n".join(help_lines)
        return ChatCompletionResponse(
            model=request.model,
            choices=[ChatCompletionChoice(message=ChatMessage(role="assistant", content=reply))],
            router_intent="help",
        )
    # ──────────────────────────────────────────────────────────────────────

    # ── Admin command: "lege benutzer an: <Name>" ─────────────────────────
    _cmd = re.match(r"^\s*lege\s+benutzer\s+an\s*:\s*(.+)$", last_message, re.IGNORECASE)
    if _cmd:
        if not authed_user_id or _get_user_role(authed_user_id) != "admin":
            reply = "⛔ Nur Administratoren können Benutzer anlegen."
        else:
            new_username = _cmd.group(1).strip()
            try:
                from db.users import create_user  # noqa: PLC0415
                new_id = create_user(new_username)
                logger.info("Admin %r created user %r (id=%r)", authed_user_id, new_username, new_id)
                reply = f'\u2705 Benutzer "{new_username}" wurde angelegt. Beim ersten Login wird ein Passwort gesetzt.'
            except Exception as exc:  # noqa: BLE001
                if "Duplicate entry" in str(exc) or "1062" in str(exc):
                    reply = f'\u26a0 Benutzer "{new_username}" existiert bereits.'
                else:
                    logger.error("create_user error: %s", exc)
                    reply = f'\u26a0 Fehler beim Anlegen: {exc}'
        return ChatCompletionResponse(
            model=request.model,
            choices=[ChatCompletionChoice(message=ChatMessage(role="assistant", content=reply))],
            router_intent="admin_command",
        )
    # ──────────────────────────────────────────────────────────────────────

    # ── Admin command: "anreicherung starten" ─────────────────────────────
    if re.match(r"^\s*anreicherung\s+starten\s*$", last_message, re.IGNORECASE):
        if not authed_user_id or _get_user_role(authed_user_id) != "admin":
            reply = "⛔ Nur Administratoren können die Anreicherung manuell starten."
        else:
            from workers.enrichment import enrich_pending_memories  # noqa: PLC0415
            background_tasks.add_task(enrich_pending_memories, llm_manager)
            reply = "🔍 Anreicherung wird im Hintergrund gestartet – gespeicherte Erinnerungen werden mit Wikipedia-Wissen verknüpft."
        return ChatCompletionResponse(
            model=request.model,
            choices=[ChatCompletionChoice(message=ChatMessage(role="assistant", content=reply))],
            router_intent="admin_command",
        )
    # ──────────────────────────────────────────────────────────────────────

    # 1. Intent classification (router LLM on P4, Phase 1: log only)
    router_result = intent_router.classify(last_message) if intent_router else None
    intent_str = router_result.intent if router_result else "general"
    logger.debug(
        "Router classification: intent=%r confidence=%s (user=%r, message=%r)",
        intent_str,
        f"{router_result.confidence:.2f}" if router_result else "n/a",
        user_id,
        last_message[:120],
    )

    # 2. Memory recall – load known facts and inject into system prompt
    from db.memory import get_pending_ambiguity, load_memories_for_prompt  # noqa: PLC0415
    memory_section = load_memories_for_prompt(user_id, last_message) if user_id != "default" else ""
    if memory_section:
        mem_lines = memory_section.count("\n") + 1
        logger.debug("Memory section: %d line(s) injected for user %r.", mem_lines, user_id)
    else:
        logger.debug("Memory section: empty for user %r.", user_id)

    # 2b. Wiki knowledge – vector-search cached Wikipedia chunks for relevant context
    wiki_section = _wiki_context_for_message(last_message)
    if wiki_section:
        logger.debug("Wiki section injected into prompt (%d chars).", len(wiki_section))
    else:
        logger.debug("Wiki section: no relevant chunks found for this message.")

    # 2c. Live Wikipedia lookup
    #
    # Priority order (mutually exclusive, first match wins):
    #   1. Correction signal → authoritative live lookup (highest priority).
    #   2. Router classified intent as "wiki" → always do live lookup, because
    #      cached chunks may be irrelevant (semantic search can return off-topic
    #      results when the topic is not yet in the local index).
    #   3. Short follow-up / clarification message → combine with the previous
    #      user turn to build the topic; do a live lookup when there is no good
    #      cached data.
    #   4. Topical question with no cached data → proactive live lookup.
    live_wiki_section = ""
    _correction_live = False  # True when live section came from a correction signal
    if _detect_correction(last_message):
        logger.info("Correction signal detected – performing live Wikipedia lookup.")
        # Build a prior-context string from the last few user messages so that
        # topic-less corrections like "Er ist gestorben" can still find the subject.
        _prior_msgs = [m.content for m in request.messages[:-1] if m.role == "user"]
        _prior_ctx = " ".join(_prior_msgs[-3:]) if _prior_msgs else ""
        live_wiki_section = await asyncio.get_running_loop().run_in_executor(
            None, _live_wiki_context_for_correction, last_message, 2, _prior_ctx
        )
        if live_wiki_section:
            _correction_live = True
            logger.info("Live wiki section injected (%d chars).", len(live_wiki_section))
        else:
            logger.info("Live wiki lookup returned no results for correction message.")
    elif _UPDATE_RE.search(last_message):
        # Benutzer möchte aktualisierte Informationen – Live-Lookup mit dem vorherigen Thema
        logger.info("Update-Signal erkannt – Live-Wikipedia-Lookup mit vorherigem Thema.")
        _prior_msgs = [m.content for m in request.messages[:-1] if m.role == "user"]
        _update_ctx = " ".join(_prior_msgs[-3:]) if _prior_msgs else last_message
        live_wiki_section = await asyncio.get_running_loop().run_in_executor(
            None, _live_wiki_context_proactive, _update_ctx
        )
        if live_wiki_section:
            logger.info("Live wiki section (Update) injected (%d chars).", len(live_wiki_section))
        else:
            logger.info("Live wiki lookup (Update) returned no results.")
    elif intent_str == "wiki":
        # Router explicitly recognised a wiki intent.  Only do a live lookup
        # when the cached data for this topic is stale (older than 7 days) or
        # does not exist yet.  Fresh cache is good enough – no extra network
        # request needed.
        wiki_topic = _extract_topic(last_message)
        is_stale = True  # default: assume stale so we do a lookup when unsure
        # "wer ist" / "who is" questions are ALWAYS treated as stale because
        # the cached article may predate a change in the current officeholder.
        if not _LIVE_ALWAYS_RE.search(last_message) and wiki_topic:
            try:
                from tools.wikipedia import wiki_topic_is_stale  # noqa: PLC0415
                is_stale = wiki_topic_is_stale(wiki_topic)
            except Exception as exc_stale:  # noqa: BLE001
                logger.warning("wiki_topic_is_stale check failed (assuming stale): %s", exc_stale)
        if is_stale:
            logger.info(
                "Wiki intent detected and cache is stale/missing for topic %r – live lookup.",
                wiki_topic,
            )
            live_wiki_section = await asyncio.get_running_loop().run_in_executor(
                None, _live_wiki_context_proactive, last_message
            )
            if live_wiki_section:
                logger.info("Live wiki section injected for wiki intent (%d chars).", len(live_wiki_section))
            else:
                logger.info("Proactive live wiki lookup (wiki intent) returned no results.")
        else:
            logger.debug(
                "Wiki intent detected but cache is fresh for topic %r – skipping live lookup.",
                wiki_topic,
            )
    else:
        # Build the effective query: if the current message is very short (likely a
        # follow-up or clarification), extend it with the last user turn from the
        # conversation history so the topic extraction has more to work with.
        _prior_user_messages = [
            m.content for m in request.messages[:-1] if m.role == "user"
        ]
        if len(last_message.split()) <= 8 and _prior_user_messages:
            _effective_query = f"{_prior_user_messages[-1]} {last_message}"
            logger.debug(
                "Short follow-up message – extended query for wiki lookup: %r",
                _effective_query[:120],
            )
        else:
            _effective_query = last_message

        # "wer ist / who is" queries ALWAYS need a live lookup — the answer may
        # have changed since the cached article was written, even if the cache is
        # technically "fresh" (e.g., the city article was cached yesterday but the
        # mayor changed a year ago and the article already reflects that).
        _live_always = bool(_LIVE_ALWAYS_RE.search(_effective_query))

        if _live_always or (not wiki_section and bool(_TOPIC_QUESTION_RE.search(_effective_query))):
            # Present-tense person question OR no cached data with a topical question
            # → proactively fetch fresh Wikipedia data.
            _reason = "present-tense person question" if _live_always else "no cached wiki data + topical question"
            logger.info("Proactive live lookup triggered (%s).", _reason)
            live_wiki_section = await asyncio.get_running_loop().run_in_executor(
                None, _live_wiki_context_proactive, _effective_query
            )
            if live_wiki_section:
                logger.info("Proactive wiki section injected (%d chars).", len(live_wiki_section))
            else:
                logger.info("Proactive live wiki lookup returned no results.")
        elif not wiki_section and len(last_message.split()) <= 8 and _prior_user_messages:
            # Short follow-up with no cached data and no question pattern either –
            # still worth a live lookup using the extended query.
            _topic = _extract_topic(_effective_query)
            if _topic:
                logger.info(
                    "Short follow-up with no cached wiki data – proactive live lookup for topic %r.",
                    _topic,
                )
                live_wiki_section = await asyncio.get_running_loop().run_in_executor(
                    None, _live_wiki_context_proactive, _effective_query
                )
                if live_wiki_section:
                    logger.info("Follow-up wiki section injected (%d chars).", len(live_wiki_section))
                else:
                    logger.info("Follow-up live wiki lookup returned no results.")

    # 2d. Proactive web search for news / sports / web_search intents
    #
    # When the router identifies a time-sensitive query (news, sports scores,
    # current events), perform a DuckDuckGo search and inject the results so
    # the LLM can answer with up-to-date information.
    web_search_section = ""
    _did_web_search = False
    if intent_str in ("news", "sports", "web_search"):
        _web_query = _extract_topic(last_message) or last_message
        _is_news = intent_str in ("news", "sports")
        logger.info(
            "Web search intent %r detected – proactive %s search for query %r.",
            intent_str, "news" if _is_news else "web", _web_query,
        )
        web_search_section = await asyncio.get_running_loop().run_in_executor(
            None, _web_search_context, _web_query, _is_news
        )
        if web_search_section:
            _did_web_search = True
            logger.info("Web search section injected (%d chars).", len(web_search_section))
        else:
            logger.info("Proactive web search returned no results for intent %r.", intent_str)

    # 2e. Proactive weather lookup for weather intent
    #
    # When the router identifies a weather query, directly call the weather
    # tool and inject the result so the LLM doesn't need to emit [WEATHER: ...]
    # tags itself (which can lead to hallucination loops).
    weather_section = ""
    if intent_str == "weather":
        _weather_location = _extract_topic(last_message) or None
        logger.info("Weather intent detected – proactive weather lookup for location %r.", _weather_location)
        try:
            from tools.weather import get_weather_for_user  # noqa: PLC0415
            weather_section = await asyncio.get_running_loop().run_in_executor(
                None, get_weather_for_user, user_id, _weather_location
            )
            if weather_section:
                logger.info("Proactive weather section injected (%d chars).", len(weather_section))
            else:
                logger.info("Proactive weather lookup returned no results.")
        except Exception as exc_weather:  # noqa: BLE001
            logger.warning("Proactive weather lookup failed (non-fatal): %s", exc_weather)

    # 3. Main LLM response (P100)
    if not llm_manager.is_ready("main"):
        return JSONResponse(
            status_code=503,
            content={"error": "Main LLM is not loaded yet. Check server logs."},
        )

    # Resolve user timezone: save to DB when provided, then always read from DB.
    from db.users import get_user_timezone, upsert_user_timezone  # noqa: PLC0415
    if request.timezone:
        try:
            ZoneInfo(request.timezone)  # validate before persisting
            upsert_user_timezone(user_id, request.timezone)
        except ZoneInfoNotFoundError:
            logger.warning("Client sent unknown timezone %r for user %r – ignored.", request.timezone, user_id)
    tz_name = get_user_timezone(user_id)
    try:
        user_tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        logger.warning("Stored timezone %r for user %r is invalid – using Europe/Berlin.", tz_name, user_id)
        user_tz = ZoneInfo("Europe/Berlin")

    # Inject current date/time (in user's timezone) into the system prompt so
    # the model can greet the user correctly (e.g. "Guten Morgen" vs "Guten Abend").
    now = datetime.now(tz=user_tz)
    now_str = now.strftime("%A, %d. %B %Y, %H:%M Uhr")
    hour = now.hour
    if 5 <= hour < 12:
        greeting = "Guten Morgen"
    elif 12 <= hour < 18:
        greeting = "Guten Tag"
    elif 18 <= hour < 22:
        greeting = "Guten Abend"
    else:
        greeting = "Hallo"
    # Only greet on the very first turn of a conversation (no prior assistant message).
    is_first_turn = not any(m.role == "assistant" for m in request.messages)
    if is_first_turn:
        time_note = (
            f"Aktuelle Uhrzeit: {now_str}. "
            f"Begrüße den Benutzer einmalig passend zur Tageszeit mit \"{greeting}\"."
        )
    else:
        time_note = f"Aktuelle Uhrzeit: {now_str}."

    # Tool-use instructions (only for authenticated non-default users)
    tool_note = ""
    if user_id != "default":
        tool_note = (
            "\n\nVerfügbare Werkzeuge (nutze sie wenn nötig, indem du sie in deiner Antwort einbettest):\n"
            "  [SEARCH_MEMORY: Suchanfrage] – Semantische Suche in gespeicherten Erinnerungen\n"
            "  [SEARCH_RELATION: Name] – Alle gespeicherten Fakten über eine bekannte Person abrufen\n"
            "  [WIKI_SEARCH: Suchanfrage] – Wikipedia on-demand abfragen (enzyklopädisches Wissen)\n"
            "  [WEB_SEARCH: Suchanfrage] – Aktuelle Websuche (Preise, Rezepte, Software, Echtzeit-Daten)\n"
            "  [NEWS_SEARCH: Suchanfrage] – Aktuelle Nachrichten und Sportergebnisse suchen\n"
            "  [WEATHER: Ort] – Aktuelles Wetter und Vorhersage abrufen (Ort optional, nutzt gespeicherten Standort)\n"
            "\n"
            "WICHTIGE REGELN für Werkzeug-Nutzung:\n"
            "1. Du hast KEINEN Zugang zu Suchmaschinen (Google, Bing etc.) und KEINEN Internetzugang.\n"
            "2. Für enzyklopädische Fakten (Personen, Orte, Geschichte, Konzepte) → [WIKI_SEARCH: ...]\n"
            "3. Für aktuelle Infos (Nachrichten, Sport, Preise, neue Releases, Echtzeit-Daten) → [WEB_SEARCH: ...] oder [NEWS_SEARCH: ...]\n"
            "4. Erfinde NIEMALS Suchergebnisse oder tue so, als hättest du im Internet gesucht.\n"
            "5. Wenn du dir bei Fakten unsicher bist, nutze ein Werkzeug BEVOR du antwortest.\n"
            "6. Bei Fragen über Personen (Wer ist X?, Was macht X?) → [WIKI_SEARCH: Name].\n"
            "7. Wenn du Informationen aus Tools verwendest, zitiere die Quelle mit einem Markdown-Link.\n"
            "8. ANTWORTE IMMER IN DER SPRACHE, IN DER DER BENUTZER SCHREIBT. "
            "Deutsch → Deutsch, Englisch → Englisch, andere Sprache → dieselbe Sprache.\n"
            "9. Wenn du Wikipedia-Quellen zitierst, formatiere sie IMMER als klickbare Markdown-Links: [Titel – Wikipedia](URL).\n"
            "10. Bilder anzeigen: Wenn im Kontext eine Zeile 'Bild: ![Titel](URL)' steht, "
            "kopiere diese Zeile EXAKT in deine Antwort – sie wird als Bild gerendert. "
            "WICHTIG: Schreibe NIEMALS 'Bild:' ohne eine vollständige Markdown-URL dahinter. "
            "NIEMALS erfundene Bild-Referenzen – nur Bilder die dir explizit im Kontext übergeben wurden."
        )
    # Inject pending disambiguation question if any
    disambig_section = ""
    if user_id != "default":
        pending_disambig = get_pending_ambiguity(user_id)
        if pending_disambig:
            q = pending_disambig.get("question", "")
            if q:
                disambig_section = (
                    f"\n\n⚠️ WICHTIG – Klärungsfrage: {q}\n"
                    "Stelle dem Benutzer genau diese Frage in deiner nächsten Antwort, "
                    "bevor du auf die aktuelle Nachricht eingehst."
                )

    # Spracherkennungs-Hinweis (für nicht-authentifizierte Benutzer ohne tool_note)
    lang_note = ""
    if user_id == "default":
        lang_note = (
            "\n\nANTWORTE IMMER IN DER SPRACHE, IN DER DER BENUTZER SCHREIBT. "
            "Deutsch → Deutsch, Englisch → Englisch, andere Sprache → dieselbe Sprache."
        )

    # Build the system prompt additions.
    # Correction live wiki goes FIRST so the model sees it before everything else.
    # Order: [correction_wiki] + time_note + memory + disambiguation + [proactive_wiki] + cached_wiki
    _WIKI_PRIORITY_NOTE = (
        "⚠️ Wikipedia-Vorrang: Wenn du Fakten aus Wikipedia-Quellen erhältst, "
        "haben diese IMMER Vorrang vor deinem eigenen Trainingswissen. "
        "Dein Trainingswissen kann veraltet sein. Wikipedia-Daten sind aktueller und vertrauenswürdiger."
    )

    if _correction_live:
        system_additions = f"{live_wiki_section}\n\n{_WIKI_PRIORITY_NOTE}\n\n{time_note}{tool_note}{lang_note}{_ANTI_HALLUCINATION_NOTE}"
        if memory_section:
            system_additions = f"{system_additions}\n\n{memory_section}"
        if disambig_section:
            system_additions = f"{system_additions}{disambig_section}"
        if wiki_section:
            system_additions = f"{system_additions}\n\n{wiki_section}"
        if web_search_section:
            system_additions = f"{system_additions}\n\n{web_search_section}"
        if weather_section:
            system_additions = f"{system_additions}\n\n{weather_section}"
    else:
        system_additions = f"{time_note}{tool_note}{lang_note}{_ANTI_HALLUCINATION_NOTE}"
        if memory_section:
            system_additions = f"{system_additions}\n\n{memory_section}"
        if disambig_section:
            system_additions = f"{system_additions}{disambig_section}"
        if live_wiki_section:
            system_additions = f"{system_additions}\n\n{_WIKI_PRIORITY_NOTE}\n\n{live_wiki_section}"
        if wiki_section:
            system_additions = f"{system_additions}\n\n{wiki_section}"
        if web_search_section:
            system_additions = f"{system_additions}\n\n{web_search_section}"
        if weather_section:
            system_additions = f"{system_additions}\n\n{weather_section}"

    messages = list(request.messages)
    if messages and messages[0].role == "system":
        messages[0] = ChatMessage(
            role="system",
            content=f"{messages[0].content}\n\n{system_additions}",
        )
    else:
        messages.insert(0, ChatMessage(role="system", content=system_additions))

    # ── Streaming path ──────────────────────────────────────────────────────
    if request.stream:
        strip = request.strip_thinking
        temperature = request.temperature
        max_tokens = request.max_tokens

        async def _sse_gen() -> "AsyncGenerator[str, None]":  # type: ignore[name-defined]
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)
            filt = _StreamThinkingFilter(strip=strip)
            collected: list[str] = []

            # Send an immediate status event so the user sees feedback while the
            # LLM is generating (especially useful when a web/wiki search was done).
            if _did_web_search:
                _status_text = "🔍 Ich habe das Web durchsucht – hier ist was ich gefunden habe:\n\n"
            elif weather_section:
                _status_text = "🌤️ Ich habe die aktuellen Wetterdaten abgerufen:\n\n"
            elif live_wiki_section or wiki_section:
                _status_text = "📚 Einen Moment, ich schaue nach...\n\n"
            else:
                _status_text = ""
            if _status_text:
                _status_payload = json.dumps(
                    {"choices": [{"index": 0, "delta": {"content": _status_text}, "finish_reason": None}]}
                )
                yield f"data: {_status_payload}\n\n"

            def _produce() -> None:
                try:
                    for raw_sse in llm_manager.chat_completion_stream(
                        "main", messages, temperature, max_tokens
                    ):
                        if raw_sse.strip() == "data: [DONE]":
                            # Flush any chars buffered by the thinking filter BEFORE
                            # forwarding [DONE].  The client breaks on [DONE], so content
                            # sent after it would be silently discarded.
                            remaining = filt.flush()
                            if remaining:
                                flush_payload = json.dumps(
                                    {"choices": [{"index": 0, "delta": {"content": remaining}, "finish_reason": None}]}
                                )
                                asyncio.run_coroutine_threadsafe(
                                    queue.put(f"data: {flush_payload}\n\n"), loop
                                ).result()
                        else:
                            # Apply thinking filter on the content delta
                            try:
                                payload = json.loads(raw_sse[len("data: "):].strip())
                                raw_content = (
                                    payload.get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                                filtered = filt.feed(raw_content)
                                payload["choices"][0]["delta"]["content"] = filtered
                                raw_sse = f"data: {json.dumps(payload)}\n\n"
                            except Exception:  # noqa: BLE001
                                pass
                        asyncio.run_coroutine_threadsafe(queue.put(raw_sse), loop).result()
                except Exception as exc:  # noqa: BLE001
                    err_payload = json.dumps({"error": str(exc)})
                    asyncio.run_coroutine_threadsafe(
                        queue.put(f"data: {err_payload}\n\n"), loop
                    ).result()
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

            threading.Thread(target=_produce, daemon=True).start()

            while True:
                item = await queue.get()
                if item is None:
                    break
                # Hold [DONE] — we inject pending images before sending it ourselves
                if item.strip() == "data: [DONE]":
                    continue
                # Collect content for post-stream background tasks
                if item.startswith("data: "):
                    try:
                        d = json.loads(item[6:].strip())
                        c = d["choices"][0]["delta"].get("content", "")
                        if c:
                            collected.append(c)
                    except Exception:  # noqa: BLE001
                        pass
                yield item

            # Server-side image injection: append any context images the LLM omitted.
            full_text = "".join(collected)
            for _img_md in _extract_pending_images(live_wiki_section or "", wiki_section or ""):
                if _img_md not in full_text:
                    _extra = f"\n\n{_img_md}"
                    _extra_payload = json.dumps({
                        "choices": [{"index": 0, "delta": {"content": _extra}, "finish_reason": None}]
                    })
                    yield f"data: {_extra_payload}\n\n"
                    full_text += _extra
                    collected.append(_extra)

            yield "data: [DONE]\n\n"

            # Fire-and-forget background work after streaming completes
            if user_id != "default" and user_id != "admin" and last_message.strip():
                from db.memory import extract_memories_sync  # noqa: PLC0415
                _recent_msgs = [m.content for m in request.messages if m.role == "user"][-4:]
                asyncio.ensure_future(
                    asyncio.to_thread(extract_memories_sync, user_id, last_message, llm_manager, _recent_msgs)
                )
            asyncio.ensure_future(
                asyncio.to_thread(_log_conversation_sync, user_id, last_message, full_text, intent_str)
            )

        return StreamingResponse(
            _sse_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # disable Nginx proxy buffering
            },
        )

    # ── Non-streaming path ───────────────────────────────────────────────────
    try:
        assistant_text = llm_manager.chat_completion(
            model_name="main",
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM inference error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"LLM inference failed: {exc}"},
        )

    # Strip <think>…</think> reasoning blocks when requested (default: True).
    if request.strip_thinking:
        assistant_text = re.sub(r"<think>.*?</think>", "", assistant_text, flags=re.DOTALL).strip()

    # Handle tool-use patterns in the assistant response.
    # If the LLM emitted tool calls (e.g. [WIKI_SEARCH: Quantenverschränkung]),
    # execute them and re-run the LLM with the enriched context (one pass only).
    tool_results = ""
    tool_calls = _parse_tool_calls(assistant_text)
    if tool_calls and user_id != "default":
        tool_results = _execute_tool_calls(tool_calls, user_id)
        if tool_results:
            logger.info("Tool-use: %d call(s) resolved, re-running LLM.", len(tool_calls))
            enriched_messages = list(messages)
            enriched_messages.append(ChatMessage(role="assistant", content=assistant_text))
            enriched_messages.append(ChatMessage(
                role="user",
                content=f"[Tool-Ergebnisse]\n{tool_results}\n\nBitte beantworte die ursprüngliche Frage nun mit diesen Informationen.",
            ))
            try:
                assistant_text = llm_manager.chat_completion(
                    model_name="main",
                    messages=enriched_messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
                if request.strip_thinking:
                    assistant_text = re.sub(r"<think>.*?</think>", "", assistant_text, flags=re.DOTALL).strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tool-use re-run failed: %s", exc)
        # Always remove tool call markers from the response shown to the user
        assistant_text = _TOOL_CALL_RE.sub("", assistant_text).strip()

    # Server-side image injection: append any context images the LLM omitted.
    _pending_imgs = _extract_pending_images(
        live_wiki_section or "", wiki_section or "", tool_results
    )
    for _img_md in _pending_imgs:
        if _img_md not in assistant_text:
            assistant_text += f"\n\n{_img_md}"

    # 4. Async memory extraction (background task, zero user latency)
    # The built-in "admin" account is excluded – it is a shared system account
    # and should not accumulate personal memories.
    if user_id != "default" and user_id != "admin" and last_message.strip():
        from db.memory import extract_memories_sync  # noqa: PLC0415
        _recent_user_msgs = [m.content for m in request.messages if m.role == "user"][-4:]
        background_tasks.add_task(
            extract_memories_sync, user_id, last_message, llm_manager, _recent_user_msgs
        )

    # 5. Log conversation to DB (fire-and-forget)
    background_tasks.add_task(
        _log_conversation_sync, user_id, last_message, assistant_text, intent_str
    )

    # 6. Return OpenAI-compatible response
    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=assistant_text)
            )
        ],
        router_intent=intent_str,
    )


@app.delete("/v1/memory/{memory_id}")
async def delete_memory_entry(memory_id: int, session_token: str | None = None) -> dict:
    """Delete a user_memory entry and its exclusively linked wiki knowledge.

    Any wiki_cache / wiki_chunks rows that were linked *only* to this memory
    are also removed (cascade).  Admins may delete any memory; regular users
    may only delete their own.

    Query parameter: ``session_token`` (required).
    """
    user_id = _resolve_token(session_token)
    if user_id is None:
        return JSONResponse(status_code=401, content={"error": "Nicht authentifiziert."})

    from db.memory import delete_memory  # noqa: PLC0415

    role = _get_user_role(user_id)
    # Admins may delete any memory; users only their own.
    owner_id = None if role == "admin" else user_id

    deleted = delete_memory(memory_id, user_id=owner_id)
    if not deleted:
        return JSONResponse(
            status_code=404,
            content={"error": f"Erinnerung {memory_id} nicht gefunden oder keine Berechtigung."},
        )
    return {"ok": True, "deleted_memory_id": memory_id}


@app.delete("/v1/relation-memory/{memory_id}")
async def delete_relation_memory_entry(memory_id: int, session_token: str | None = None) -> dict:
    """Delete a relation_memory entry.

    Admins may delete any entry; regular users may only delete their own.

    Query parameter: ``session_token`` (required).
    """
    user_id = _resolve_token(session_token)
    if user_id is None:
        return JSONResponse(status_code=401, content={"error": "Nicht authentifiziert."})

    from db.relations import delete_relation_memory  # noqa: PLC0415

    role = _get_user_role(user_id)
    owner_id = None if role == "admin" else user_id

    deleted = delete_relation_memory(memory_id, user_id=owner_id)
    if not deleted:
        return JSONResponse(
            status_code=404,
            content={"error": f"Beziehungserinnerung {memory_id} nicht gefunden oder keine Berechtigung."},
        )
    return {"ok": True, "deleted_memory_id": memory_id}


# ---------------------------------------------------------------------------
# Admin API endpoints for manual worker triggers
# ---------------------------------------------------------------------------

@app.get("/api/image/{image_id}")
async def serve_image(image_id: int, thumb: bool = False) -> StreamingResponse:
    """Liefert ein gecachtes Bild anhand seiner ID.

    ?thumb=true gibt das WebP-Vorschaubild zurück (falls vorhanden).
    Gibt 404 zurück, wenn das Bild nicht gefunden wurde.
    """
    from db.images import get_image  # noqa: PLC0415

    record = get_image(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Bild nicht gefunden.")

    if thumb and record.get("thumb_data"):
        data = record["thumb_data"]
        media_type = "image/webp"
    else:
        data = record.get("image_data")
        if not data:
            raise HTTPException(status_code=404, detail="Bilddaten nicht verfügbar.")
        media_type = record.get("mime_type", "application/octet-stream")

    return Response(
        content=bytes(data),
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.post("/admin/cleanup")
async def trigger_cleanup(session_token: str | None = None) -> dict:
    """Manually trigger the cleanup worker (admin only)."""
    user_id = _resolve_token(session_token)
    if user_id is None:
        return JSONResponse(status_code=401, content={"error": "Nicht authentifiziert."})
    if _get_user_role(user_id) != "admin":
        return JSONResponse(status_code=403, content={"error": "Nur Administratoren können den Cleanup starten."})
    from workers.cleanup import run_cleanup_now  # noqa: PLC0415
    loop = asyncio.get_running_loop()
    summary = await loop.run_in_executor(None, run_cleanup_now, llm_manager)
    return {"ok": True, "summary": summary}


@app.post("/admin/enrichment")
async def trigger_enrichment(session_token: str | None = None) -> dict:
    """Manually trigger the enrichment worker (admin only)."""
    user_id = _resolve_token(session_token)
    if user_id is None:
        return JSONResponse(status_code=401, content={"error": "Nicht authentifiziert."})
    if _get_user_role(user_id) != "admin":
        return JSONResponse(status_code=403, content={"error": "Nur Administratoren können die Anreicherung starten."})
    from workers.enrichment import enrich_pending_memories  # noqa: PLC0415
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, enrich_pending_memories, llm_manager)
    return {"ok": True, "message": "Anreicherung abgeschlossen."}



_WEB_DIR = _HERE / "web"
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="static")


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn  # noqa: PLC0415

    try:
        import config as _cfg  # noqa: PLC0415
        host = getattr(_cfg, "HOST", "0.0.0.0")
        port = getattr(_cfg, "PORT", 8000)
    except ImportError:
        host, port = "0.0.0.0", 8000

    # SSL-Zertifikat automatisch erzeugen (selbstsigniert, 10 Jahre)
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None
    try:
        from tools.ssl_cert import ensure_ssl_cert  # noqa: PLC0415
        cert_paths = ensure_ssl_cert()
        if cert_paths:
            ssl_certfile, ssl_keyfile = cert_paths
    except Exception as _ssl_exc:  # noqa: BLE001
        logger.warning("SSL-Zertifikat konnte nicht erstellt werden: %s", _ssl_exc)

    if ssl_certfile and ssl_keyfile:
        logger.info("Starte ice_brain mit HTTPS auf %s:%s", host, port)
        uvicorn.run(
            "server:app",
            host=host,
            port=port,
            reload=False,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
        )
    else:
        logger.info("Starte ice_brain mit HTTP (kein SSL) auf %s:%s", host, port)
        uvicorn.run("server:app", host=host, port=port, reload=False)
