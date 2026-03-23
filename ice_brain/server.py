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

import logging
import os
import sys
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
from fastapi.responses import JSONResponse
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

    # 1. Intent classification (router LLM on P4, Phase 1: log only)
    router_result = intent_router.classify(last_message) if intent_router else None
    intent_str = router_result.intent if router_result else "general"

    # 2. Memory recall – load known facts and inject into system prompt
    from db.memory import load_memories_for_prompt  # noqa: PLC0415
    memory_section = load_memories_for_prompt(user_id) if user_id != "default" else ""

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
    time_note = (
        f"Aktuelle Uhrzeit: {now_str}. "
        f"Begrüße den Benutzer passend zur Tageszeit mit \"{greeting}\"."
    )
    # Build the system prompt additions: time note + memory section
    system_additions = time_note
    if memory_section:
        system_additions = f"{system_additions}\n\n{memory_section}"

    messages = list(request.messages)
    if messages and messages[0].role == "system":
        messages[0] = ChatMessage(
            role="system",
            content=f"{messages[0].content}\n\n{system_additions}",
        )
    else:
        messages.insert(0, ChatMessage(role="system", content=system_additions))

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


# ---------------------------------------------------------------------------
# Static WebUI (must be mounted AFTER routes)
# ---------------------------------------------------------------------------
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
