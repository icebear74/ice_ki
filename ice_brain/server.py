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
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="ice_brain", version="0.1.0")

# Globals – populated during startup
llm_manager: LLMManager = LLMManager()
intent_router: IntentRouter | None = None
_server_tz: ZoneInfo = ZoneInfo("Europe/Berlin")  # overridden from config at startup


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    global _server_tz  # noqa: PLW0603
    # 1. Load config
    try:
        import config  # noqa: PLC0415
        models_cfg = config.MODELS
        tz_name = getattr(config, "TIMEZONE", "Europe/Berlin")
        try:
            _server_tz = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            logger.warning("Unknown TIMEZONE %r in config – falling back to Europe/Berlin.", tz_name)
    except ImportError:
        logger.error(
            "config.py not found!  Run:  cp config.py.example config.py  "
            "and fill in your model paths and MySQL credentials."
        )
        models_cfg = {}

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

    logger.info("ice_brain ready.  Model status: %s", llm_manager.get_status())


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


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
) -> ChatCompletionResponse:
    user_id = request.user or "default"
    last_message = request.messages[-1].content if request.messages else ""

    # 1. Intent classification (router LLM on P4, Phase 1: log only)
    router_result = intent_router.classify(last_message) if intent_router else None
    intent_str = router_result.intent if router_result else "general"

    # 2. Phase 1: no RAG / memory lookup – proceed directly to main LLM

    # 3. Main LLM response (P100)
    if not llm_manager.is_ready("main"):
        return JSONResponse(
            status_code=503,
            content={"error": "Main LLM is not loaded yet. Check server logs."},
        )

    # Inject current date/time into the system prompt so the model can greet
    # the user correctly (e.g. "Guten Morgen" vs "Guten Abend").
    now = datetime.now()
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
    messages = list(request.messages)
    if messages and messages[0].role == "system":
        messages[0] = ChatMessage(
            role="system",
            content=f"{messages[0].content}\n\n{time_note}",
        )
    else:
        messages.insert(0, ChatMessage(role="system", content=time_note))

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

    # 4. Phase 1: no async memory extraction

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
