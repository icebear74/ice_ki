"""
User Memory – extraction, storage and recall.

Two memory tiers
----------------
- Short-term  : has an expires_at timestamp (activity, mood, plan, topic)
- Long-term   : expires_at IS NULL – permanent (preference, personal,
                relationship, hobby, experience)

Extraction runs as a BACKGROUND TASK after every assistant response so the
user sees zero added latency.  The main model (8B) is used for extraction
to ensure high-quality fact parsing.

Public API
----------
extract_memories_sync(user_id, user_message, llm_manager)
    Call from a background thread.  Uses router LLM to extract facts,
    deduplicates against existing rows and upserts into user_memory.

load_memories_for_prompt(user_id) -> str
    Return a formatted string (German) ready to be appended to the
    system prompt.  Returns "" when there are no memories.
"""

from __future__ import annotations

import difflib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_manager import LLMManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Categories and their default TTLs
# ---------------------------------------------------------------------------

# Category → default TTL in hours (None = permanent)
_CATEGORY_TTL: dict[str, int | None] = {
    "preference":   None,   # permanent
    "personal":     None,   # permanent
    "relationship": None,   # permanent
    "hobby":        None,   # permanent
    "experience":   None,   # permanent
    "activity":     2,      # expires in 2 h
    "mood":         6,      # expires in 6 h
    "plan":         48,     # expires in 48 h
    "topic":        1,      # expires in 1 h
    "timezone":     None,   # managed separately – never overwritten here
}

_VALID_CATEGORIES = set(_CATEGORY_TTL.keys())

# Tuneable thresholds
_MAX_MEMORIES_FOR_PROMPT = 20       # memories injected into system prompt
_MAX_SIMILARITY_SEARCH_ROWS = 50    # rows scanned for deduplication per category
_MIN_WORD_LENGTH_FOR_SIMILARITY = 4 # minimum word length used for similarity matching
_MIN_MESSAGE_LENGTH = 4             # messages shorter than this are not extracted

_TTL_MAP: dict[str, int] = {
    "1h": 1,
    "2h": 2,
    "4h": 4,
    "8h": 8,
    "24h": 24,
    "48h": 48,
}

# ---------------------------------------------------------------------------
# Extraction prompt (English for better JSON output)
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """\
You are a memory extraction assistant. Your job is to extract personal facts \
about the user from their message and output them as a JSON array.

Rules:
- Extract ALL facts – a single message can contain multiple facts.
- Write facts in THIRD PERSON (e.g. "Drinks coffee", not "I drink coffee").
- ONLY output the JSON array – no prose, no markdown fences.
- If the message contains NO extractable personal facts (e.g. "ok", "thanks", \
"what is the weather?"), output an empty array: []
- IMPORTANT: Also extract facts from INDIRECT or QUESTION-FORM statements. \
If the user says "did you know I like coffee?" or "wusstest du dass ich Kaffee mag?", \
this still reveals the fact "Likes coffee" – extract it!

DISCOURSE MARKERS – do NOT extract these as facts:
Words like "ich denke", "ich glaube", "ich meine", "ich vermute", "I think", \
"I believe", "I suppose", "I guess" are hedging phrases – they introduce the \
REAL statement that follows. Never store "thinks about X" or "believes X". \
Instead extract the actual fact from the rest of the sentence.
  Example: "ich denke, meine Kaffeemaschine kann das gut" → fact: owns a coffee machine.

WEATHER / ENVIRONMENT – do NOT extract these as personal facts:
Observations about weather, temperature, nature, or the environment (e.g. \
"Es wird wärmer", "Das Wetter wird besser", "Es regnet", "It's getting warmer") \
describe the world, NOT the user. Never store them as mood, activity, or any \
other personal fact. Skip them entirely.

POSSESSIVES = OWNERSHIP – always extract ownership as a permanent fact:
Words like "mein", "meine", "my", "die ich habe", "die ich besitze", "ich besitze", \
"I own", "I have", "I possess", "at home", "zu hause" reveal that the user OWNS \
or holds a membership/pass/subscription for something.
  → Extract as category=personal, ttl=null.
  → Content: "Besitzt [object]" / "Hat [membership/pass] für [place]".
  → This includes passes, memberships, subscriptions (Jahreskarte, Abo, annual pass).

FUTURE WISH / "es wird Zeit" = hobby or preference signal:
Phrases like "es wird Zeit mal wieder X zu tun", "I should really do X again", \
"höchste Zeit für X" mean the user already knows and enjoys X – it reveals a \
PERMANENT habit or preference/hobby.
  → Extract as category=hobby or preference, ttl=null.
  → Do NOT add a short-term activity entry – it hasn't happened yet.

HABITUAL vs. CURRENT – this distinction is critical:

HABITUAL signals (words like: gerne, manchmal, ab und zu, oft, öfters, immer, \
häufig, meistens, abends, morgens, jeden Tag, sometimes, usually, often, \
always, like to, love to, tend to):
  → The statement describes a PERMANENT habit or preference.
  → Extract ONE permanent fact only: category=preference or hobby, ttl=null.
  → Do NOT add a short-term activity entry – there is no "right now" meaning.

CURRENT signals (words like: gerade, jetzt, im Moment, gerade eben, soeben, \
right now, currently, at the moment, just now):
  → The statement describes something happening RIGHT NOW.
  → Extract TWO facts:
      1. The current action:  category=activity, ttl appropriate (e.g. "2h")
      2. The underlying habit: category=preference or hobby, ttl=null
  → Both facts are needed: one is temporary, one is permanent.
  → If the current activity involves another person (e.g. "Lisa kocht gerade"), \
also infer a permanent preference/habit for that person (e.g. "Lisa kocht gerne").

Valid categories: preference, personal, relationship, hobby, experience, \
activity, mood, plan, topic

Valid ttl values: "1h", "2h", "4h", "8h", "24h", "48h", or null (permanent)

JSON schema for each fact:
{
  "content": "<fact in third person, concise>",
  "category": "<one of the valid categories>",
  "importance": <float 0.0-1.0>,
  "ttl": "<ttl string or null>"
}

Examples:

Message: "Ich trinke gerne abends mal einen Espresso"
→ "gerne" + "mal" are HABITUAL signals → permanent preference only, NO activity with ttl
Output:
[
  {"content": "Trinkt gerne abends Espresso", "category": "preference", "importance": 0.7, "ttl": null}
]

Message: "Ich trinke gerade einen Espresso"
→ "gerade" is a CURRENT signal → activity (short-term) + preference (permanent)
Output:
[
  {"content": "Trinkt gerade Espresso", "category": "activity", "importance": 0.3, "ttl": "2h"},
  {"content": "Trinkt Espresso / mag Espresso", "category": "preference", "importance": 0.6, "ttl": null}
]

Message: "Ich trinke gerade einen Kaffee"
→ "gerade" is a CURRENT signal → activity (short-term) + preference (permanent)
Output:
[
  {"content": "Trinkt gerade Kaffee", "category": "activity", "importance": 0.3, "ttl": "2h"},
  {"content": "Trinkt Kaffee / mag Kaffee", "category": "preference", "importance": 0.6, "ttl": null}
]

Message: "wusstest du dass ich kaffee mag?"
Output:
[
  {"content": "Mag Kaffee", "category": "preference", "importance": 0.6, "ttl": null}
]

Message: "did you know I love hiking?"
Output:
[
  {"content": "Loves hiking", "category": "hobby", "importance": 0.7, "ttl": null}
]

Message: "Ich war im August im Urlaub"
Output:
[
  {"content": "War im August im Urlaub", "category": "experience", "importance": 0.5, "ttl": null}
]

Message: "Meine Frau Lisa kocht gerade Pasta"
→ "gerade" is a CURRENT signal → activity (short-term) + preference (permanent)
Output:
[
  {"content": "Lisa kocht gerade Pasta", "category": "activity", "importance": 0.3, "ttl": "2h"},
  {"content": "Hat eine Frau namens Lisa", "category": "relationship", "importance": 0.8, "ttl": null},
  {"content": "Lisa kocht gerne", "category": "preference", "importance": 0.5, "ttl": null}
]

Message: "Ich bin müde, war heute den ganzen Tag joggen"
Output:
[
  {"content": "Ist gerade müde", "category": "mood", "importance": 0.4, "ttl": "4h"},
  {"content": "War heute joggen", "category": "activity", "importance": 0.3, "ttl": "24h"},
  {"content": "Geht joggen / ist sportlich", "category": "hobby", "importance": 0.7, "ttl": null}
]

Message: "Ich gehe manchmal abends spazieren"
→ "manchmal" is a HABITUAL signal → permanent hobby only
Output:
[
  {"content": "Geht manchmal abends spazieren", "category": "hobby", "importance": 0.5, "ttl": null}
]

Message: "ich denke, meine Siemens EQ9 S700 die ich zu hause habe, kann das schon ganz gut"
→ "ich denke" is a DISCOURSE MARKER – ignore it, extract the real facts.
→ "meine … die ich zu hause habe" → OWNS the machine → permanent personal fact.
→ "kann das schon ganz gut" → positive opinion about it → permanent preference fact.
Output:
[
  {"content": "Besitzt eine Siemens EQ9 S700 Kaffeemaschine", "category": "personal", "importance": 0.8, "ttl": null},
  {"content": "Findet die Siemens EQ9 S700 gut / ist zufrieden damit", "category": "preference", "importance": 0.6, "ttl": null}
]

Message: "Das Wetter wird besser, es wird Zeit mal wieder Achterbahn zu fahren, zb im Moviepark wo ich ja eine Jahreskarte besitze"
→ "Das Wetter wird besser" → WEATHER observation → skip, not a personal fact.
→ "es wird Zeit mal wieder Achterbahn zu fahren" → FUTURE WISH signal → permanent hobby.
→ "eine Jahreskarte besitze" + "Moviepark" → POSSESSIVES/OWNERSHIP → permanent personal fact.
Output:
[
  {"content": "Fährt gerne Achterbahn", "category": "hobby", "importance": 0.7, "ttl": null},
  {"content": "Besitzt eine Jahreskarte für den Moviepark", "category": "personal", "importance": 0.8, "ttl": null}
]

Message: "ok"
Output:
[]
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_facts(raw: str) -> list[dict]:
    """Parse the JSON array from the LLM response.  Returns [] on failure."""
    raw = raw.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    # Sometimes the model wraps the array in an outer key
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find a JSON array in the output
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                logger.warning("Memory extraction: could not parse JSON from model output: %r", raw[:200])
                return []
        else:
            logger.warning("Memory extraction: no JSON array found in model output: %r", raw[:200])
            return []

    if not isinstance(data, list):
        return []
    return data


def _compute_expires_at(ttl_str: str | None, category: str) -> datetime | None:
    """Return the expiry datetime for a memory, or None for permanent."""
    if ttl_str is None:
        return None
    hours = _TTL_MAP.get(ttl_str)
    if hours is None:
        # Fall back to category default
        default_hours = _CATEGORY_TTL.get(category)
        if default_hours is None:
            return None
        hours = default_hours
    return datetime.now(tz=timezone.utc) + timedelta(hours=hours)


def _normalise_fact(fact: dict) -> dict | None:
    """Validate and normalise a raw fact dict.  Returns None to skip."""
    content = str(fact.get("content", "")).strip()
    if not content:
        return None
    category = str(fact.get("category", "")).strip().lower()
    if category not in _VALID_CATEGORIES or category == "timezone":
        category = "preference"  # safe fallback
    try:
        importance = float(fact.get("importance", 0.5))
        importance = max(0.0, min(1.0, importance))
    except (TypeError, ValueError):
        importance = 0.5
    ttl_str = fact.get("ttl")
    if ttl_str is not None and str(ttl_str) not in _TTL_MAP:
        ttl_str = None
    return {
        "content": content,
        "category": category,
        "importance": importance,
        "ttl": ttl_str,
    }


# ---------------------------------------------------------------------------
# DB read / write
# ---------------------------------------------------------------------------

def _find_similar(cursor, user_id: str, category: str, content: str) -> int | None:
    """Return the id of an existing row that is similar to *content*, or None.

    Two-stage matching:
    1. Fast word-based substring check: the stored content contains one of the
       first two meaningful words (≥4 chars) of the new content.
    2. Fuzzy fallback via difflib.SequenceMatcher: ratio() ≥ 0.75 counts as a
       match so that typos like "Phantasieland" vs "Phantasialand" are caught.
    """
    cursor.execute(
        "SELECT id, content FROM user_memory "
        "WHERE user_id = %s AND category = %s "
        "AND (expires_at IS NULL OR expires_at > NOW()) "
        "LIMIT %s",
        (user_id, category, _MAX_SIMILARITY_SEARCH_ROWS),
    )
    rows = cursor.fetchall()
    if not rows:
        return None

    # Use first meaningful word (>= 4 chars) of new content as key
    words = [w.lower() for w in content.split() if len(w) >= _MIN_WORD_LENGTH_FOR_SIMILARITY]

    content_lower = content.lower()

    for row_id, row_content in rows:
        row_lower = row_content.lower()

        # Stage 1: fast word-based substring matching
        if words:
            for word in words[:2]:
                if word in row_lower:
                    return row_id

        # Stage 2: fuzzy full-content comparison
        ratio = difflib.SequenceMatcher(None, content_lower, row_lower).ratio()
        if ratio >= 0.75:
            return row_id

    return None


def _upsert_fact(cursor, user_id: str, fact: dict) -> None:
    """Insert or update a single fact in user_memory."""
    expires_at = _compute_expires_at(fact["ttl"], fact["category"])
    existing_id = _find_similar(cursor, user_id, fact["category"], fact["content"])

    if existing_id is not None:
        if expires_at is not None:
            cursor.execute(
                "UPDATE user_memory SET content = %s, importance = %s, "
                "updated_at = NOW(), expires_at = %s "
                "WHERE id = %s",
                (fact["content"], fact["importance"], expires_at, existing_id),
            )
        else:
            cursor.execute(
                "UPDATE user_memory SET content = %s, importance = %s, "
                "updated_at = NOW(), expires_at = NULL "
                "WHERE id = %s",
                (fact["content"], fact["importance"], existing_id),
            )
        logger.debug(
            "Memory updated (id=%d, category=%s): %s",
            existing_id, fact["category"], fact["content"]
        )
    else:
        cursor.execute(
            "INSERT INTO user_memory (user_id, category, content, importance, expires_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (user_id, fact["category"], fact["content"], fact["importance"], expires_at),
        )
        logger.debug(
            "Memory inserted (category=%s): %s", fact["category"], fact["content"]
        )


# ---------------------------------------------------------------------------
# Public: extraction (runs in background thread)
# ---------------------------------------------------------------------------

def extract_memories_sync(user_id: str, user_message: str, llm_manager: "LLMManager") -> None:
    """Extract facts from *user_message* and persist them for *user_id*.

    Designed to run in a background thread (via FastAPI BackgroundTasks or
    asyncio.get_event_loop().run_in_executor).  Never raises – all errors are
    logged.
    """
    if not user_message or not user_message.strip():
        return

    # Skip trivially short messages
    if len(user_message.strip()) < _MIN_MESSAGE_LENGTH:
        return

    try:
        if not llm_manager.is_ready("main"):
            logger.info("Memory extraction skipped – main model not loaded.")
            return

        from models import ChatMessage  # noqa: PLC0415

        messages = [
            ChatMessage(role="system", content=_EXTRACTION_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_message),
        ]

        raw = llm_manager.chat_completion(
            model_name="main",
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )

        facts_raw = _parse_facts(raw)
        if not facts_raw:
            logger.info(
                "Memory extraction: no facts extracted for user %r. "
                "Model output: %r",
                user_id, raw[:200],
            )
            return

        facts = [_normalise_fact(f) for f in facts_raw]
        facts = [f for f in facts if f is not None]

        if not facts:
            return

        from db.connection import get_connection  # noqa: PLC0415

        with get_connection() as conn:
            cursor = conn.cursor()
            stored = 0
            for fact in facts:
                try:
                    _upsert_fact(cursor, user_id, fact)
                    stored += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not upsert memory fact: %s", exc)
            conn.commit()
            cursor.close()

        logger.info(
            "Memory extraction: %d fact(s) stored for user %r.", stored, user_id
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("Memory extraction failed for user %r: %s", user_id, exc)


# ---------------------------------------------------------------------------
# Public: recall (called before main LLM)
# ---------------------------------------------------------------------------

def load_memories_for_prompt(user_id: str) -> str:
    """Return a German-language memory section for the system prompt.

    Returns "" when there are no relevant memories or on any error.
    """
    try:
        from db.connection import get_connection  # noqa: PLC0415

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT category, content, importance, expires_at "
                "FROM user_memory "
                "WHERE user_id = %s "
                "  AND category != 'timezone' "
                "  AND (expires_at IS NULL OR expires_at > NOW()) "
                "ORDER BY importance DESC, updated_at DESC "
                "LIMIT %s",
                (user_id, _MAX_MEMORIES_FOR_PROMPT),
            )
            rows = cursor.fetchall()
            cursor.close()

        if not rows:
            return ""

        permanent: list[str] = []
        temporary: list[str] = []

        for category, content, importance, expires_at in rows:
            entry = f"- {content} [{category}]"
            if expires_at is None:
                permanent.append(entry)
            else:
                temporary.append(entry)

        parts: list[str] = ["Bekannte Informationen über den Benutzer:"]
        if permanent:
            parts.append("\n📌 Dauerhaft:")
            parts.extend(permanent)
        if temporary:
            parts.append("\n⏳ Aktuell:")
            parts.extend(temporary)

        return "\n".join(parts)

    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load memories for user %r: %s", user_id, exc)
        return ""
