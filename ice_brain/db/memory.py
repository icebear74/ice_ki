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
import re
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
    "location":     None,   # permanent – where user lives / works / is located
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

# Generic high-frequency words that appear as the first word in many memory entries
# and must NOT be used alone as a similarity key (they would cause false duplicates).
# Example: "Möchte mit 'Sir' angesprochen werden" vs.
#          "Möchte wie ein ungehobelter Mensch angesprochen werden"
# both start with "Möchte" – without this guard they would incorrectly be treated
# as duplicates and the second would overwrite the first.
_SIMILARITY_STOP_WORDS: frozenset[str] = frozenset({
    # German verbs / modal / auxiliary commonly used as memory entry prefixes
    "möchte", "mag", "magst", "will", "will,", "muss", "kann", "soll",
    "hat", "hatte", "haben", "ist", "war", "sind", "waren",
    "wird", "wurde", "wäre", "geht", "fährt", "macht", "trinkt",
    "isst", "spielt", "liebt", "hasst", "kennt", "wohnt", "lebt",
    "arbeitet", "schläft", "liest", "hört", "sieht", "sucht",
    # English equivalents (in case mixed-language entries occur)
    "likes", "loves", "hates", "wants", "needs", "has", "have",
    "is", "was", "are", "were", "does", "will", "would", "prefers",
})

_TTL_MAP: dict[str, int] = {
    "1h": 1,
    "2h": 2,
    "4h": 4,
    "8h": 8,
    "12h": 12,
    "24h": 24,
    "48h": 48,
    "72h": 72,
    "96h": 96,
    "120h": 120,
    "168h": 168,
}

# ---------------------------------------------------------------------------
# Extraction prompt (English for better JSON output)
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """\
You are a memory extraction assistant. Extract personal facts from the user's message and output a JSON object.

CRITICAL RULES:
- Every fact MUST have an explicit "subject": "user" for facts about the user, or the person's name (e.g. "susanne") for facts about third parties.
- "relation_type" is REQUIRED when subject is not "user" (partner/friend/family/colleague/acquaintance/other).
- Write facts in THIRD PERSON (e.g. "Drinks coffee", not "I drink coffee").
- Output ONLY the JSON – no prose, no markdown fences.
- "temporal": "permanent" for lasting facts, "current" for right-now, "past" for one-time past events.
- "ttl": null for permanent, "2h"/"24h"/"48h" for temporary.
- Also extract facts from INDIRECT or QUESTION-FORM statements. "wusstest du dass ich Kaffee mag?" still reveals "Mag Kaffee" – extract it.
- Possessives ("mein", "meine", "my") reveal ownership/preferences – always extract them.

ANTI-HALLUCINATION:
- Extract ONLY facts EXPLICITLY stated by the user. NEVER infer or invent details like years, model years, model numbers, prices, quantities, or specifications that the user did NOT mention.
- If the user says "Ich fahre einen Opel Corsa", extract ONLY "Fährt einen Opel Corsa" – do NOT add a model year, engine size, or any other detail not stated.
- If the user says "Ich habe ein Auto", extract only "Hat ein Auto" – do NOT add a brand, model, or year.

PRONOUN RESOLUTION:
- "ihre/sein/ihren/seinen/ihrem/seinem/ihrer/seines" (her/his) → fact about the LAST MENTIONED PERSON, NOT the user.
- "unser/unsere/unserem/unseren" (our/shared) → extract for BOTH the user AND the mentioned relation person.
- "mein/meine/meinen/meinem" (my) → user fact only.

CORRECTIONS:
- "nicht nur X, sondern Y" / "eigentlich Y, nicht X" / "sie ist nicht X, sie ist Y" → use Y as the correct value.
  Extract the corrected type for that person, discard the wrong label.

RELATION FACTS – NAME IN CONTENT:
- Content of relation facts MUST include the person's name.
  BAD: "Kocht gerne"           →  GOOD: "Melanie kocht gerne"
  BAD: "Hat Kinder"            →  GOOD: "Melanie hat Kinder namens Timo und Sarah"
  BAD: "Ist Kind des Benutzers" →  GOOD: subject="emily", content="Emily ist Kind des Benutzers"

OWNERSHIP / POSSESSION:
- "auf unserem X" / "unser X" / "unsere X" → user owns X AND the mentioned relation person owns X (shared).
  Extract a personal fact for the user AND a personal fact for the relation person.

DO NOT EXTRACT:
- Weather/environment observations ("Es regnet", "Es wird wärmer").
- Third-party states attributed to the user ("Susanne is sick" must NOT be stored as user fact).
- AI corrections about third parties ("Er ist gestorben" = THIRD PARTY FACT, NOT user fact).
- Pure information requests with no personal disclosure.
- IGNORE code-specific implementation details: variable names, function signatures, API endpoints, pin assignments, buffer sizes, class names, method names.
- Only extract PERSONAL facts about the user: tools/languages they use, projects they work on, hardware they own.
- Do NOT store code snippets, technical implementation details, or debugging context as personal facts.
- "I use an ESP32 with PSRAM" → EXTRACT (personal tool/hardware fact)
- "The buffer is 16KB" → IGNORE (implementation detail)
- "I want SPI on pin 18" → IGNORE (implementation detail)

LOCATION FACTS:
- Categorize where the user lives, works, or is located as "location", NOT "preference".
- For temporary stays ("Ich bin 3 Tage in Berlin", "Ich fahre morgen nach Paris"), set "ttl" to the appropriate duration (e.g. "72h" for 3 days).
- For same-day visits ("Ich bin heute im Phantasialand", "Ich bin grade auf dem Nürburgring"), set "ttl" to "24h" and temporal to "current".
- "heute" / "today" → ttl "24h". "grade" / "gerade" / "momentan" → ttl "8h".
- POI (Points of Interest) like theme parks, stadiums, landmarks count as locations too:
  "Ich bin im Phantasialand" → category "location", content "Ist im Phantasialand", ttl "8h", temporal "current".
  "Ich bin heute auf dem Nürburgring" → category "location", content "Ist heute auf dem Nürburgring", ttl "24h", temporal "current".
- For permanent moves ("Ich bin nach München gezogen") or residence ("Ich wohne in Köln"), set "temporal" to "permanent" with no TTL and category "location".
- Do NOT include latitude/longitude fields in the JSON output – geocoding is handled automatically after extraction.
- Examples: "Ich wohne in Dinslaken" → category "location", content "Wohnt in Dinslaken".

AMBIGUITY: When a relation type is unclear (e.g. "meine Freundin" could be partner or friend), add an entry to "ambiguities".

JSON schema:
{
  "facts": [
    {
      "subject": "user",
      "content": "<fact in third person>",
      "category": "<preference|personal|relationship|hobby|experience|location|activity|mood|plan|topic>",
      "importance": <0.0-1.0>,
      "temporal": "<permanent|current|past>",
      "ttl": "<null or 1h/2h/4h/8h/12h/24h/48h/72h/96h/120h/168h>"
    },
    {
      "subject": "susanne",
      "relation_type": "friend",
      "content": "<Name + fact about susanne in third person>",
      "category": "personal",
      "importance": 0.6,
      "temporal": "permanent",
      "ttl": null
    }
  ],
  "ambiguities": [
    {
      "question": "<German follow-up question to ask user>",
      "context": "<why this is ambiguous>"
    }
  ]
}

Examples:

Message: "Susanne habe ich in die Firma gebracht, sie arbeitet als DevOps Engineer wie ich"
Output:
{"facts": [
  {"subject": "user", "content": "Arbeitet als DevOps Engineer", "category": "personal", "importance": 0.8, "temporal": "permanent", "ttl": null},
  {"subject": "susanne", "relation_type": "acquaintance", "content": "Susanne arbeitet als DevOps Engineer", "category": "personal", "importance": 0.6, "temporal": "permanent", "ttl": null},
  {"subject": "susanne", "relation_type": "acquaintance", "content": "Susanne arbeitet in derselben Firma wie User", "category": "personal", "importance": 0.6, "temporal": "permanent", "ttl": null}
], "ambiguities": []}

Message: "Meine Bekannte Susanne legt sich hin, die ist krank"
Output:
{"facts": [
  {"subject": "susanne", "relation_type": "acquaintance", "content": "Susanne ist gerade krank", "category": "mood", "importance": 0.5, "temporal": "current", "ttl": "24h"}
], "ambiguities": []}

Message: "Ich trinke gerne abends mal einen Espresso"
Output:
{"facts": [
  {"subject": "user", "content": "Trinkt gerne abends Espresso", "category": "preference", "importance": 0.7, "temporal": "permanent", "ttl": null}
], "ambiguities": []}

Message: "Du solltest wissen, das Melanie nicht nur Freundin, sondern meine Partnerin ist, mit der ich zusammen wohne"
Output:
{"facts": [
  {"subject": "user", "content": "Lebt mit Partnerin Melanie zusammen", "category": "personal", "importance": 0.8, "temporal": "permanent", "ttl": null},
  {"subject": "melanie", "relation_type": "partner", "content": "Melanie ist die Partnerin des Benutzers", "category": "relationship", "importance": 0.9, "temporal": "permanent", "ttl": null},
  {"subject": "melanie", "relation_type": "partner", "content": "Melanie wohnt mit dem Benutzer zusammen", "category": "personal", "importance": 0.7, "temporal": "permanent", "ttl": null}
], "ambiguities": []}

Message: "Wir darten gerne auf unserem Dartautomaten"
Output:
{"facts": [
  {"subject": "user", "content": "Spielt gerne Dart", "category": "hobby", "importance": 0.7, "temporal": "permanent", "ttl": null},
  {"subject": "user", "content": "Besitzt einen Dartautomaten", "category": "personal", "importance": 0.6, "temporal": "permanent", "ttl": null}
], "ambiguities": []}

Message: "Ihre Kinder heißen Timo und Sarah, meine heißen Emily und Tony"
Output:
{"facts": [
  {"subject": "user", "content": "Hat Kinder namens Emily und Tony", "category": "relationship", "importance": 0.9, "temporal": "permanent", "ttl": null},
  {"subject": "emily", "relation_type": "family", "content": "Emily ist Kind des Benutzers", "category": "personal", "importance": 0.8, "temporal": "permanent", "ttl": null},
  {"subject": "tony", "relation_type": "family", "content": "Tony ist Kind des Benutzers", "category": "personal", "importance": 0.8, "temporal": "permanent", "ttl": null}
], "ambiguities": [{"question": "Wessen Kinder sind Timo und Sarah? Sind das die Kinder deiner Partnerin?", "context": "'ihre Kinder' – Pronomen bezieht sich auf eine zuvor genannte Person"}]}

Message: "Ich wohne in Dinslaken"
Output:
{"facts": [
  {"subject": "user", "content": "Wohnt in Dinslaken", "category": "location", "importance": 0.9, "temporal": "permanent", "ttl": null}
], "ambiguities": []}

Message: "Ich bin heute im Phantasialand"
Output:
{"facts": [
  {"subject": "user", "content": "Ist heute im Phantasialand", "category": "location", "importance": 0.7, "temporal": "current", "ttl": "24h"}
], "ambiguities": []}

Message: "Ich bin grade für 2 Tage in Berlin"
Output:
{"facts": [
  {"subject": "user", "content": "Ist für 2 Tage in Berlin", "category": "location", "importance": 0.7, "temporal": "current", "ttl": "48h"}
], "ambiguities": []}

Message: "ok"
Output:
{"facts": [], "ambiguities": []}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_extraction_result(raw: str) -> tuple[list[dict], list[dict]]:
    """Parse the JSON extraction result from the LLM.

    Returns (facts, ambiguities).  Handles both the new object schema
    {"facts": [...], "ambiguities": [...]} and the legacy flat array [...].
    """
    raw = raw.strip()
    # Strip <think>…</think> reasoning blocks emitted by some models.
    raw = re.sub(r"<think>.*?(?:</think>|$)", "", raw, flags=re.DOTALL)
    raw = raw.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    # Try to parse JSON
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find a JSON object or array in the output
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = raw.find(start_char)
            end = raw.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(raw[start:end + 1])
                    break
                except json.JSONDecodeError:
                    continue
        if data is None:
            logger.warning("Memory extraction: could not parse JSON: %r", raw[:200])
            return [], []

    # New schema: {"facts": [...], "ambiguities": [...]}
    if isinstance(data, dict):
        facts = data.get("facts", [])
        ambiguities = data.get("ambiguities", [])
        if not isinstance(facts, list):
            facts = []
        if not isinstance(ambiguities, list):
            ambiguities = []
        return facts, ambiguities

    # Legacy schema: flat array of facts (no subject field)
    if isinstance(data, list):
        # Inject subject="user" for backward compat
        for f in data:
            if isinstance(f, dict) and "subject" not in f:
                f["subject"] = "user"
        return data, []

    return [], []


def _parse_facts(raw: str) -> list[dict]:
    """Parse the JSON array from the LLM response.  Returns [] on failure.

    Legacy helper kept for backward compatibility.
    """
    facts, _ = _parse_extraction_result(raw)
    return facts


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
    temporal = str(fact.get("temporal", "permanent"))
    if temporal not in ("permanent", "current", "past"):
        temporal = "permanent"
    return {
        "content": content,
        "category": category,
        "importance": importance,
        "ttl": ttl_str,
        "temporal": temporal,
    }


def _normalise_fact_with_subject(fact: dict) -> dict | None:
    """Same as _normalise_fact but preserves subject/relation_type fields."""
    norm = _normalise_fact(fact)
    if norm is None:
        return None
    norm["subject"] = str(fact.get("subject", "user")).strip().lower()
    norm["relation_type"] = str(fact.get("relation_type", "acquaintance")).strip().lower()
    return norm


# ---------------------------------------------------------------------------
# Pending disambiguation cache (in-process, cleared on restart)
# ---------------------------------------------------------------------------

# user_id → list of pending questions
_pending_disambiguations: dict[str, list[dict]] = {}


def _store_pending_ambiguity(user_id: str, ambiguities: list[dict]) -> None:
    """Store disambiguation questions for the next LLM turn."""
    if not ambiguities:
        return
    if user_id not in _pending_disambiguations:
        _pending_disambiguations[user_id] = []
    for a in ambiguities:
        if isinstance(a, dict) and a.get("question"):
            _pending_disambiguations[user_id].append(a)
    logger.debug(
        "Stored %d pending disambiguation(s) for user %r.", len(ambiguities), user_id
    )


def get_pending_ambiguity(user_id: str) -> dict | None:
    """Pop and return the oldest pending disambiguation question, or None."""
    questions = _pending_disambiguations.get(user_id)
    if not questions:
        return None
    return questions.pop(0)


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

    For ``location`` facts, also scans the ``preference`` category to catch old
    location facts that were stored under the wrong category (migration path).
    """
    # For location facts, also check 'preference' rows so we can overwrite
    # old location facts that were previously mis-categorised.
    categories_to_search = [category]
    if category == "location":
        categories_to_search.append("preference")

    rows: list[tuple] = []
    for cat in categories_to_search:
        cursor.execute(
            "SELECT id, content FROM user_memory "
            "WHERE user_id = %s AND category = %s "
            "AND (expires_at IS NULL OR expires_at > NOW()) "
            "LIMIT %s",
            (user_id, cat, _MAX_SIMILARITY_SEARCH_ROWS),
        )
        rows.extend(cursor.fetchall())
    if not rows:
        return None

    # Use first two meaningful words (>= 4 chars, not generic stop-words) as keys.
    # Stop-words like "Möchte", "Hat", "Trinkt" are intentionally excluded: they
    # appear as the first word in many entries of the same category and would cause
    # unrelated facts (e.g. two distinct preferences) to be falsely deduplicated.
    words = [
        w.lower()
        for w in content.split()
        if len(w) >= _MIN_WORD_LENGTH_FOR_SIMILARITY
        and w.lower().rstrip(",.;:!?") not in _SIMILARITY_STOP_WORDS
    ]

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


def _upsert_fact(cursor, user_id: str, fact: dict) -> int | None:
    """Insert or update a single fact in user_memory (with embedding on write).

    Returns the row ID of the affected row (existing_id on UPDATE, lastrowid on INSERT),
    or None if the operation failed silently.
    """
    expires_at = _compute_expires_at(fact["ttl"], fact["category"])
    existing_id = _find_similar(cursor, user_id, fact["category"], fact["content"])

    temporal = str(fact.get("temporal", "permanent"))
    if temporal not in ("permanent", "current", "past"):
        temporal = "permanent"

    # Compute embedding for semantic recall
    embedding_text: str | None = None
    try:
        from tools.embeddings import embed_one, vec_to_text  # noqa: PLC0415
        vec = embed_one(fact["content"])
        embedding_text = vec_to_text(vec)
    except Exception as exc:  # noqa: BLE001
        logger.debug("_upsert_fact: embedding failed (non-fatal): %s", exc)

    if existing_id is not None:
        # Update the existing row; also correct the category in case an old
        # fact was stored under the wrong category (e.g. location → preference).
        if embedding_text:
            if expires_at is not None:
                cursor.execute(
                    "UPDATE user_memory SET content = %s, category = %s, importance = %s, temporal = %s, "
                    "updated_at = NOW(), expires_at = %s, embedding = VEC_FromText(%s) "
                    "WHERE id = %s",
                    (fact["content"], fact["category"], fact["importance"], temporal, expires_at, embedding_text, existing_id),
                )
            else:
                cursor.execute(
                    "UPDATE user_memory SET content = %s, category = %s, importance = %s, temporal = %s, "
                    "updated_at = NOW(), expires_at = NULL, embedding = VEC_FromText(%s) "
                    "WHERE id = %s",
                    (fact["content"], fact["category"], fact["importance"], temporal, embedding_text, existing_id),
                )
        else:
            if expires_at is not None:
                cursor.execute(
                    "UPDATE user_memory SET content = %s, category = %s, importance = %s, temporal = %s, "
                    "updated_at = NOW(), expires_at = %s "
                    "WHERE id = %s",
                    (fact["content"], fact["category"], fact["importance"], temporal, expires_at, existing_id),
                )
            else:
                cursor.execute(
                    "UPDATE user_memory SET content = %s, category = %s, importance = %s, temporal = %s, "
                    "updated_at = NOW(), expires_at = NULL "
                    "WHERE id = %s",
                    (fact["content"], fact["category"], fact["importance"], temporal, existing_id),
                )
        logger.debug(
            "Memory updated (id=%d, category=%s): %s",
            existing_id, fact["category"], fact["content"]
        )
        return existing_id
    else:
        if embedding_text:
            cursor.execute(
                "INSERT INTO user_memory (user_id, category, content, importance, temporal, expires_at, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, VEC_FromText(%s))",
                (user_id, fact["category"], fact["content"], fact["importance"],
                 temporal, expires_at, embedding_text),
            )
        else:
            cursor.execute(
                "INSERT INTO user_memory (user_id, category, content, importance, temporal, expires_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (user_id, fact["category"], fact["content"], fact["importance"], temporal, expires_at),
            )
        new_id = cursor.lastrowid
        logger.debug(
            "Memory inserted (id=%s, category=%s): %s", new_id, fact["category"], fact["content"]
        )
        return new_id


# ---------------------------------------------------------------------------
# Public: extraction (runs in background thread)
# ---------------------------------------------------------------------------

def delete_memory(memory_id: int, user_id: str | None = None) -> bool:
    """Delete a user_memory row and clean up exclusively linked wiki knowledge.

    If *user_id* is given the deletion is scoped to that user (ownership check).

    Cascade behaviour:
    - wiki_cache entries linked *only* to this memory (not to any other) are
      deleted together with their wiki_chunks rows.
    - memory_knowledge_link rows are cleaned up automatically via FK CASCADE.

    Returns True on success, False when the row was not found or not owned.
    """
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()

            # Verify the row exists (and belongs to user_id when given)
            if user_id is not None:
                cursor.execute(
                    "SELECT id FROM user_memory WHERE id = %s AND user_id = %s",
                    (memory_id, user_id),
                )
            else:
                cursor.execute(
                    "SELECT id FROM user_memory WHERE id = %s",
                    (memory_id,),
                )
            if cursor.fetchone() is None:
                cursor.close()
                logger.info("delete_memory: id=%d not found (or wrong user).", memory_id)
                return False

            # Find wiki_cache IDs that are exclusively linked to this memory
            # (i.e., not linked to any other memory row).
            cursor.execute(
                "SELECT mkl.cache_id "
                "FROM memory_knowledge_link mkl "
                "WHERE mkl.memory_id = %s "
                "AND NOT EXISTS ("
                "    SELECT 1 FROM memory_knowledge_link mkl2 "
                "    WHERE mkl2.cache_id = mkl.cache_id AND mkl2.memory_id != %s"
                ")",
                (memory_id, memory_id),
            )
            orphan_cache_ids = [row[0] for row in cursor.fetchall()]

            # Delete wiki_chunks and wiki_cache for orphaned articles
            if orphan_cache_ids:
                placeholders = ",".join(["%s"] * len(orphan_cache_ids))
                cursor.execute(
                    f"DELETE FROM wiki_chunks WHERE article_id IN ({placeholders})",  # noqa: S608
                    orphan_cache_ids,
                )
                cursor.execute(
                    f"DELETE FROM wiki_cache WHERE id IN ({placeholders})",  # noqa: S608
                    orphan_cache_ids,
                )
                logger.info(
                    "delete_memory: removed %d orphaned wiki_cache entr%s for memory id=%d.",
                    len(orphan_cache_ids),
                    "y" if len(orphan_cache_ids) == 1 else "ies",
                    memory_id,
                )

            # Delete the memory row (memory_knowledge_link CASCADE-deleted by FK)
            cursor.execute("DELETE FROM user_memory WHERE id = %s", (memory_id,))
            conn.commit()
            cursor.close()

        logger.info("delete_memory: memory id=%d deleted.", memory_id)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("delete_memory error for id=%d: %s", memory_id, exc)
        return False


def _geocode_location_fact(user_id: str, content: str, memory_id: int) -> None:
    """Geocode a location fact and embed coordinates directly into the content text.

    Extracts a place name from *content* (e.g. "Wohnt in Dinslaken" → "Dinslaken"),
    calls the geocoding tool, enriches the content string with display_name and
    coordinates in the format "Wohnt in Dinslaken, Nordrhein-Westfalen (📍 51.5672, 6.7331)",
    then UPDATEs the DB row (content + embedding).

    Idempotent: if *content* already contains '📍', nothing is done.
    Non-fatal – errors are silently swallowed so they never block memory extraction.
    """
    try:
        # Already geocoded – skip to stay idempotent
        if "📍" in content:
            return

        # Extract the place name using extended German preposition patterns
        match = re.search(
            r"(?:in|im|nach|bei|aus|von|auf\s+de[mnr]|am|"
            r"wohn(?:t|en)\s+in|lebt?\s+in|zog\s+nach|"
            r"[Ii]st\s+(?:heute\s+)?(?:im|in|auf\s+de[mnr]|am|bei))\s+"
            r"([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+(?:[\s\-][A-Za-zÄÖÜäöüß\-]+)*)",
            content,
        )
        if not match:
            # Fallback: take the last capitalised word-sequence as the place name
            caps = re.findall(r"\b[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+)*", content)
            place_name = caps[-1].strip() if caps else ""
        else:
            place_name = match.group(1).strip()

        if not place_name:
            return

        from tools.geocoding import geocode  # noqa: PLC0415
        result = geocode(place_name)
        if not result:
            logger.debug(
                "Geocoding returned no result for place %r (user %r).",
                place_name, user_id,
            )
            return

        lat = result["lat"]
        lon = result["lon"]
        display_name = result.get("display_name", place_name)

        # Build enriched content: append display_name and coordinates
        # Use only the first two components of the display_name (e.g. "Dinslaken, Nordrhein-Westfalen")
        display_parts = [p.strip() for p in display_name.split(",")]
        short_display = ", ".join(display_parts[:2])

        # Avoid redundancy if the place name is already part of the content
        if short_display.lower() in content.lower():
            enriched_content = f"{content.rstrip()} (📍 {lat:.4f}, {lon:.4f})"
        else:
            enriched_content = f"{content.rstrip()}, {short_display} (📍 {lat:.4f}, {lon:.4f})"

        # Compute new embedding for enriched content
        embedding_text: str | None = None
        try:
            from tools.embeddings import embed_one, vec_to_text  # noqa: PLC0415
            vec = embed_one(enriched_content)
            embedding_text = vec_to_text(vec)
        except Exception as emb_exc:  # noqa: BLE001
            logger.debug("_geocode_location_fact: embedding failed (non-fatal): %s", emb_exc)

        # UPDATE the DB row with enriched content and new embedding
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            if embedding_text:
                cursor.execute(
                    "UPDATE user_memory SET content = %s, updated_at = NOW(), "
                    "embedding = VEC_FromText(%s) WHERE id = %s",
                    (enriched_content, embedding_text, memory_id),
                )
            else:
                cursor.execute(
                    "UPDATE user_memory SET content = %s, updated_at = NOW() WHERE id = %s",
                    (enriched_content, memory_id),
                )
            conn.commit()
            cursor.close()

        logger.info(
            "Geocoded location fact for user %r (id=%d): %r → lat=%.4f lon=%.4f (%s)",
            user_id, memory_id, place_name, lat, lon, display_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("_geocode_location_fact failed (non-fatal): %s", exc)


def extract_memories_sync(
    user_id: str,
    user_message: str,
    llm_manager: "LLMManager",
    recent_messages: list[str] | None = None,
) -> None:
    """Extract facts from *user_message* and persist them for *user_id*.

    Designed to run in a background thread (via FastAPI BackgroundTasks or
    asyncio.get_event_loop().run_in_executor).  Never raises – all errors are
    logged.

    *recent_messages* is an optional list of the last few user messages (oldest
    first) that provides context for correction detection.  When supplied, the
    extraction prompt receives the recent conversation history so the LLM can
    detect statements like "Nein, ich meine NRW, nicht Rheinland-Pfalz".

    Facts with subject="user" are upserted into user_memory.
    Facts with subject="<name>" are routed to relation_memory via relations.
    Ambiguities are stored in the pending_disambiguations in-memory cache so
    the main LLM can ask the user a follow-up question on the next turn.
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

        # Build the extraction input: include recent conversation history when
        # available so the LLM can detect corrections like "eigentlich NRW".
        if recent_messages and len(recent_messages) > 1:
            history_lines = "\n".join(
                f"[Frühere Nachricht] {m}" for m in recent_messages[:-1]
            )
            extraction_input = (
                f"{history_lines}\n"
                f"[Aktuelle Nachricht] {user_message}"
            )
        else:
            extraction_input = user_message

        # Append /no_think to suppress Qwen3's extended reasoning block.
        messages = [
            ChatMessage(role="system", content=_EXTRACTION_SYSTEM_PROMPT),
            ChatMessage(role="user", content=f"{extraction_input}\n/no_think"),
        ]

        raw = llm_manager.chat_completion(
            model_name="main",
            messages=messages,
            temperature=0.0,
            max_tokens=4096,
        )

        facts_raw, ambiguities = _parse_extraction_result(raw)
        if not facts_raw and not ambiguities:
            logger.info(
                "Memory extraction: no facts extracted for user %r. "
                "Model output: %r",
                user_id, raw[:200],
            )
            return

        # Store ambiguities for follow-up question on next turn
        if ambiguities:
            _store_pending_ambiguity(user_id, ambiguities)

        # Partition facts into user-facts and relation-facts
        user_facts: list[dict] = []
        relation_facts: list[tuple[str, str, dict]] = []  # (name, relation_type, fact)

        for raw_fact in facts_raw:
            if not isinstance(raw_fact, dict):
                continue
            subject = str(raw_fact.get("subject", "user")).strip().lower()
            if subject == "user":
                norm = _normalise_fact(raw_fact)
                if norm:
                    user_facts.append(norm)
            else:
                # Fact about a named person
                relation_type = str(raw_fact.get("relation_type", "acquaintance")).strip().lower()
                norm = _normalise_fact_with_subject(raw_fact)
                if norm:
                    relation_facts.append((subject, relation_type, norm))

        stored = 0

        # ── Store user facts ─────────────────────────────────────────────────
        if user_facts:
            from db.connection import get_connection  # noqa: PLC0415
            location_facts_with_ids: list[tuple[dict, int | None]] = []
            with get_connection() as conn:
                cursor = conn.cursor()
                for fact in user_facts:
                    try:
                        fact_id = _upsert_fact(cursor, user_id, fact)
                        stored += 1
                        if fact.get("category") == "location":
                            location_facts_with_ids.append((fact, fact_id))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Could not upsert memory fact: %s", exc)
                conn.commit()
                cursor.close()

            # ── Geocode location facts ─────────────────────────────────────
            # Geocode ALL location facts (permanent and temporary) to embed
            # coordinates directly into the content text (no schema change needed).
            for fact, fact_id in location_facts_with_ids:
                if fact_id is not None:
                    _geocode_location_fact(user_id, fact["content"], fact_id)

        # ── Store relation facts ─────────────────────────────────────────────
        if relation_facts:
            from db.relations import find_or_create_relation, upsert_relation_fact  # noqa: PLC0415
            for name, relation_type, fact in relation_facts:
                try:
                    relation_id, created = find_or_create_relation(
                        user_id, name, relation_type
                    )
                    upsert_relation_fact(relation_id, fact)
                    stored += 1
                    if created:
                        logger.info(
                            "New relation created: %r (%s) for user %r.",
                            name, relation_type, user_id,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not upsert relation fact for %r: %s", name, exc)

        logger.info(
            "Memory extraction: %d fact(s) stored for user %r (ambiguities: %d).",
            stored, user_id, len(ambiguities),
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("Memory extraction failed for user %r: %s", user_id, exc)


# ---------------------------------------------------------------------------
# Public: semantic recall
# ---------------------------------------------------------------------------

_SEMANTIC_RECALL_LIMIT = 5   # top-K results from semantic search
_SEMANTIC_MIN_SCORE = 0.30   # minimum cosine similarity to include


def semantic_recall(user_id: str, user_message: str, limit: int = _SEMANTIC_RECALL_LIMIT) -> list[dict]:
    """Semantic vector search across user_memory, relation_memory, and wiki_chunks.

    Returns up to *limit* results sorted by score descending.
    Each result has: source ('user_memory'|'relation_memory'|'wiki'), content,
    category, score, and optional extra fields.

    Returns [] on error or when no embeddings are available.
    """
    if not user_message or not user_message.strip():
        return []

    try:
        from tools.embeddings import cosine_similarity, embed_one, unpack_embedding  # noqa: PLC0415
        query_vec = embed_one(user_message)
    except Exception as exc:  # noqa: BLE001
        logger.warning("semantic_recall: embedding failed: %s", exc)
        return []

    candidates: list[tuple[float, dict]] = []

    # ── user_memory (excluding is_core and timezone) ─────────────────────────
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, category, content, importance, embedding "
                "FROM user_memory "
                "WHERE user_id = %s AND is_core = FALSE AND category != 'timezone' "
                "AND (expires_at IS NULL OR expires_at > NOW()) "
                "AND embedding IS NOT NULL",
                (user_id,),
            )
            rows = cursor.fetchall()
            cursor.close()
        for row_id, category, content, importance, emb_raw in rows:
            try:
                vec = unpack_embedding(emb_raw)
                score = cosine_similarity(query_vec, vec)
                candidates.append((score, {
                    "source": "user_memory",
                    "id": row_id,
                    "category": category,
                    "content": content,
                    "importance": importance,
                }))
            except Exception:  # noqa: BLE001
                continue
    except Exception as exc:  # noqa: BLE001
        logger.debug("semantic_recall: user_memory query failed: %s", exc)

    # ── relation_memory ───────────────────────────────────────────────────────
    try:
        from db.relations import semantic_search_relation_memory  # noqa: PLC0415
        rel_results = semantic_search_relation_memory(user_id, query_vec, limit=limit)
        for r in rel_results:
            candidates.append((r["score"], {
                "source": "relation_memory",
                "id": r["id"],
                "relation_id": r["relation_id"],
                "name": r["name"],
                "relation_type": r["relation_type"],
                "category": r["category"],
                "content": r["content"],
                "importance": r["importance"],
            }))
    except Exception as exc:  # noqa: BLE001
        logger.debug("semantic_recall: relation_memory search failed: %s", exc)

    # ── wiki_chunks ───────────────────────────────────────────────────────────
    try:
        from db.wiki import search_wiki_chunks  # noqa: PLC0415
        wiki_results = search_wiki_chunks(user_message, limit=limit)
        for r in wiki_results:
            if r.get("score", 0) >= _SEMANTIC_MIN_SCORE:
                candidates.append((r["score"], {
                    "source": "wiki",
                    "id": r["id"],
                    "category": "knowledge",
                    "content": r["content"],
                    "title": r.get("title", ""),
                    "importance": r.get("score", 0.5),
                }))
    except Exception as exc:  # noqa: BLE001
        logger.debug("semantic_recall: wiki search failed: %s", exc)

    # Sort and filter
    candidates.sort(key=lambda x: x[0], reverse=True)
    results = [
        r for score, r in candidates[:limit]
        if score >= _SEMANTIC_MIN_SCORE
    ]
    return results


# ---------------------------------------------------------------------------
# Public: recall (called before main LLM)
# ---------------------------------------------------------------------------

def load_memories_for_prompt(user_id: str, user_message: str = "") -> str:
    """Return a German-language memory section for the system prompt.

    1. Core memories (is_core=TRUE) – always injected.
    2. Semantic recall (vector search) for top-5 contextually relevant memories.

    Returns "" when there are no relevant memories or on any error.
    """
    try:
        from db.connection import get_connection  # noqa: PLC0415

        # ── 1. Core memories (always inject) ──────────────────────────────
        core_entries: list[str] = []
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT category, content FROM user_memory "
                "WHERE user_id = %s AND is_core = TRUE "
                "AND category != 'timezone' "
                "AND (expires_at IS NULL OR expires_at > NOW()) "
                "ORDER BY importance DESC",
                (user_id,),
            )
            rows = cursor.fetchall()
            cursor.close()
        for category, content in rows:
            core_entries.append(f"- {content} [{category}]")

        # ── 2. Semantic recall for contextual memories ─────────────────────
        semantic_entries: list[str] = []
        if user_message:
            recalled = semantic_recall(user_id, user_message)
            for item in recalled:
                source = item["source"]
                content = item["content"]
                category = item.get("category", "")
                if source == "relation_memory":
                    name = item.get("name", "")
                    rel_type = item.get("relation_type", "")
                    semantic_entries.append(f"- [{name} ({rel_type})] {content} [{category}]")
                elif source == "wiki":
                    title = item.get("title", "")
                    semantic_entries.append(f"- [Wiki: {title}] {content[:200]}")
                else:
                    semantic_entries.append(f"- {content} [{category}]")
        else:
            # Fallback: top memories by importance when no message available
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT category, content, importance, expires_at "
                    "FROM user_memory "
                    "WHERE user_id = %s AND is_core = FALSE "
                    "  AND category != 'timezone' "
                    "  AND (expires_at IS NULL OR expires_at > NOW()) "
                    "ORDER BY importance DESC, updated_at DESC "
                    "LIMIT %s",
                    (user_id, _MAX_MEMORIES_FOR_PROMPT),
                )
                rows = cursor.fetchall()
                cursor.close()
            for category, content, importance, expires_at in rows:
                entry = f"- {content} [{category}]"
                semantic_entries.append(entry)

        if not core_entries and not semantic_entries:
            return ""

        parts: list[str] = ["Bekannte Informationen über den Benutzer:"]
        if core_entries:
            parts.append("\n📌 Grundregeln (immer beachten):")
            parts.extend(core_entries)
        if semantic_entries:
            parts.append("\n🧠 Relevante Erinnerungen:")
            parts.extend(semantic_entries)

        return "\n".join(parts)

    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load memories for user %r: %s", user_id, exc)
        return ""
