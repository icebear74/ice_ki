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

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("ice_brain")

# ---------------------------------------------------------------------------
# App + session store
# ---------------------------------------------------------------------------
app = FastAPI(title="ice_brain", version="0.1.0")

# token → user_id  (in-memory; cleared on restart)
_sessions: dict[str, str] = {}

# Globals – populated during startup
llm_manager: LLMManager = LLMManager()
intent_router: IntentRouter | None = None


def _new_token(user_id: str) -> str:
    token = secrets.token_hex(32)
    _sessions[token] = user_id
    return token


def _resolve_token(token: str | None) -> str | None:
    """Return user_id for *token*, or None if invalid/missing."""
    if not token:
        return None
    return _sessions.get(token)


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
# Startup / Shutdown
# ---------------------------------------------------------------------------

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
async def auth_login(req: LoginRequest) -> LoginResponse:
    from db.users import authenticate  # noqa: PLC0415
    from db.connection import get_connection  # noqa: PLC0415

    result = authenticate(req.username, req.password)
    if result is None:
        return JSONResponse(status_code=401, content={"error": "Ungültiger Benutzername oder Passwort."})

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

    token = None if first_login else _new_token(user_id)
    return LoginResponse(
        user_id=user_id,
        username=req.username,
        role=role,
        first_login=first_login,
        token=token,
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
# Correction detection + live Wikipedia lookup
# ---------------------------------------------------------------------------

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
    r"|was\s+(?:ist|sind|war|waren|bedeutet|weiß)"
    r"|wer\s+(?:ist|war)"
    r"|wo\s+(?:ist|liegt|befindet|gibt)"
    r"|wie\s+(?:ist|war|funktioniert|geht)"
    r"|can\s+you\s+tell"
    r"|do\s+you\s+know"
    r"|what\s+is"
    r"|gibt\s+es"
    r"|handelt\s+(?:es\s+)?sich"
    r"|erkl(?:är|aer)"
    r"|\?\s*$"
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
                lines.append(f"  Quelle: {r['source_url']}")
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
    logger.info("Proactive live wiki lookup (no cached data). Query: %r", topic)
    try:
        from tools.wikipedia import wiki_live_lookup  # noqa: PLC0415
        results = wiki_live_lookup(topic, limit=limit)
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
                lines.append(f"  Quelle: {r['source_url']}")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Proactive live wiki lookup failed (non-fatal): %s", exc)
        return ""




class _StreamThinkingFilter:
    """State machine that removes ``<think>…</think>`` blocks from a token
    stream when *strip* is True.

    Feed tokens one at a time via :meth:`feed`; call :meth:`flush` after the
    last token to drain any bytes buffered for boundary detection.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self, strip: bool) -> None:
        self._strip = strip
        self._buf: str = ""
        self._in_think: bool = False

    def feed(self, token: str) -> str:  # noqa: C901
        if not self._strip:
            return token
        self._buf += token
        out_parts: list[str] = []
        while self._buf:
            if self._in_think:
                idx = self._buf.find(self._CLOSE)
                if idx >= 0:
                    self._buf = self._buf[idx + len(self._CLOSE):]
                    self._in_think = False
                else:
                    keep = len(self._CLOSE) - 1
                    self._buf = self._buf[-keep:] if len(self._buf) > keep else self._buf
                    break
            else:
                idx = self._buf.find(self._OPEN)
                if idx >= 0:
                    out_parts.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(self._OPEN):]
                    self._in_think = True
                else:
                    keep = len(self._OPEN) - 1
                    safe = len(self._buf) - keep
                    if safe > 0:
                        out_parts.append(self._buf[:safe])
                        self._buf = self._buf[safe:]
                    break
        return "".join(out_parts)

    def flush(self) -> str:
        """Return any buffered content held for boundary detection."""
        if self._strip and self._in_think:
            self._buf = ""
            return ""
        out = self._buf
        self._buf = ""
        return out


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





def _wiki_context_for_message(message: str, limit: int = 3, min_score: float = 0.35) -> str:
    """Search cached wiki chunks for *message* and format as a prompt section.

    Returns an empty string when no relevant chunks are found, when the
    embedding model is not yet loaded, or on any error (non-fatal).
    """
    if not message or len(message.strip()) < 4:
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
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Wiki context search failed (non-fatal): %s", exc)
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
    from db.memory import load_memories_for_prompt  # noqa: PLC0415
    memory_section = load_memories_for_prompt(user_id) if user_id != "default" else ""
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
    elif intent_str == "wiki":
        # Router explicitly recognised a wiki intent.  Only do a live lookup
        # when the cached data for this topic is stale (older than 7 days) or
        # does not exist yet.  Fresh cache is good enough – no extra network
        # request needed.
        wiki_topic = _extract_topic(last_message)
        is_stale = True  # default: assume stale so we do a lookup when unsure
        if wiki_topic:
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
        if not wiki_section and bool(_TOPIC_QUESTION_RE.search(_effective_query)):
            # No cached data and the effective query looks like a topical question →
            # proactively fetch fresh Wikipedia data.
            logger.info("No cached wiki data and topical question detected – proactive live lookup.")
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
    # Build the system prompt additions.
    # Correction live wiki goes FIRST so the model sees it before everything else.
    # Order: [correction_wiki] + time_note + memory + [proactive_wiki] + cached_wiki
    if _correction_live:
        system_additions = f"{live_wiki_section}\n\n{time_note}"
        if memory_section:
            system_additions = f"{system_additions}\n\n{memory_section}"
        if wiki_section:
            system_additions = f"{system_additions}\n\n{wiki_section}"
    else:
        system_additions = time_note
        if memory_section:
            system_additions = f"{system_additions}\n\n{memory_section}"
        if live_wiki_section:
            system_additions = f"{system_additions}\n\n{live_wiki_section}"
        if wiki_section:
            system_additions = f"{system_additions}\n\n{wiki_section}"

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
                # Collect content for post-stream background tasks
                if item.strip() != "data: [DONE]" and item.startswith("data: "):
                    try:
                        d = json.loads(item[6:].strip())
                        c = d["choices"][0]["delta"].get("content", "")
                        if c:
                            collected.append(c)
                    except Exception:  # noqa: BLE001
                        pass
                yield item
            yield "data: [DONE]\n\n"

            # Fire-and-forget background work after streaming completes
            full_text = "".join(collected)
            if user_id != "default" and user_id != "admin" and last_message.strip():
                from db.memory import extract_memories_sync  # noqa: PLC0415
                asyncio.ensure_future(
                    asyncio.to_thread(extract_memories_sync, user_id, last_message, llm_manager)
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

    # 4. Async memory extraction (background task, zero user latency)
    # The built-in "admin" account is excluded – it is a shared system account
    # and should not accumulate personal memories.
    if user_id != "default" and user_id != "admin" and last_message.strip():
        from db.memory import extract_memories_sync  # noqa: PLC0415
        background_tasks.add_task(
            extract_memories_sync, user_id, last_message, llm_manager
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

    uvicorn.run("server:app", host=host, port=port, reload=False)
