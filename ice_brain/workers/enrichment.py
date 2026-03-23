"""
Enrichment-Worker – Background Knowledge Enrichment Loop.

Für jeden unangereicherten user_memory-Eintrag (enriched=FALSE) mit
relevanter Kategorie und ausreichender Wichtigkeit:
1. Main-LLM gibt Wikipedia-Suchbegriffe vor (JSON-Array, max 3)
2. wikipedia.wiki_search() wird für jeden Begriff aufgerufen
3. Treffer werden in memory_knowledge_link verknüpft
4. Eintrag wird als enriched=TRUE markiert
5. keywords-Feld im wiki_cache wird vom Main-LLM befüllt

Der Worker läuft als asyncio-Background-Task, prüft alle 30 Minuten und
überspringt den Lauf falls das Main-LLM gerade belegt ist.

Konfiguration
-------------
MAX_ENRICHMENTS_PER_RUN     – wie viele Einträge pro Durchlauf verarbeitet werden
MAX_WIKI_QUERIES_PER_FACT   – maximale Wikipedia-Abfragen pro Fakt
MIN_IMPORTANCE_TO_ENRICH    – minimale Wichtigkeit für Anreicherung
ENRICHMENT_INTERVAL_SECONDS – Pause zwischen zwei Läufen (Standard: 30 Minuten)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_manager import LLMManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
MAX_ENRICHMENTS_PER_RUN: int = 5
MAX_WIKI_QUERIES_PER_FACT: int = 3
MAX_WIKI_RESULTS_PER_TERM: int = 3  # Wikipedia results fetched per search term
MIN_IMPORTANCE_TO_ENRICH: float = 0.5
ENRICHMENT_INTERVAL_SECONDS: int = 30 * 60  # 30 Minuten

_ENRICHABLE_CATEGORIES = {"preference", "hobby", "personal", "experience"}

_SEARCH_TERMS_PROMPT = """\
You are a knowledge assistant. Given a personal fact about a user, output a \
JSON array of up to {max_terms} Wikipedia search terms (in German) that are \
most relevant for enriching that fact with background knowledge.

Rules:
- Output ONLY the JSON array – no prose, no markdown fences.
- Each term should be a concise German search query (1-4 words).
- If no Wikipedia enrichment makes sense, output an empty array: []

Fact: {fact}
"""

_KEYWORDS_PROMPT = """\
Extract 5-10 concise German keywords from the following Wikipedia summary that \
describe its main topics. Output them as a comma-separated list only – no prose.

Summary: {summary}
"""


# ---------------------------------------------------------------------------
# Core enrichment logic (synchronous, runs in a thread)
# ---------------------------------------------------------------------------

def enrich_pending_memories(llm_manager: "LLMManager") -> None:
    """Load pending memories and enrich them with Wikipedia knowledge.

    Designed to be called from a background thread / asyncio executor.
    Never raises – all errors are logged.
    """
    if not llm_manager.is_ready("main"):
        logger.info("Enrichment skipped – main model not loaded.")
        return

    # Check if the main model lock is currently held (busy during inference).
    # We do a non-blocking acquire; if it fails the model is in use.
    from llm_manager import _ModelHandle  # noqa: PLC0415 (internal, same package)
    handle = llm_manager._models.get("main")  # noqa: SLF001
    if handle is not None and not handle.lock.acquire(blocking=False):
        logger.info("Enrichment skipped – main model is busy.")
        return
    if handle is not None:
        handle.lock.release()

    try:
        _run_enrichment(llm_manager)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Enrichment loop error: %s", exc)


def _run_enrichment(llm_manager: "LLMManager") -> None:
    """Fetch pending memories and enrich them."""
    from db.connection import get_connection  # noqa: PLC0415
    from tools.wikipedia import wiki_search  # noqa: PLC0415

    with get_connection() as conn:
        cursor = conn.cursor()
        placeholders = ",".join(["%s"] * len(_ENRICHABLE_CATEGORIES))
        cursor.execute(
            f"SELECT id, content, category FROM user_memory "  # noqa: S608
            f"WHERE enriched = FALSE "
            f"AND category IN ({placeholders}) "
            f"AND importance >= %s "
            f"AND (expires_at IS NULL OR expires_at > NOW()) "
            f"LIMIT %s",
            (*_ENRICHABLE_CATEGORIES, MIN_IMPORTANCE_TO_ENRICH, MAX_ENRICHMENTS_PER_RUN),
        )
        rows = cursor.fetchall()
        cursor.close()

    if not rows:
        logger.info("Enrichment: no pending memories found.")
        return

    logger.info("Enrichment: processing %d memory entr%s.", len(rows), "y" if len(rows) == 1 else "ies")

    for memory_id, content, category in rows:
        success = False
        try:
            success = _enrich_single(llm_manager, memory_id, content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Enrichment failed for memory id=%d: %s", memory_id, exc)
            continue

        if not success:
            continue

        # Mark as enriched only when the LLM+Wikipedia step completed without error
        try:
            from db.connection import get_connection as _gc  # noqa: PLC0415
            with _gc() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE user_memory SET enriched = TRUE, enriched_at = NOW() WHERE id = %s",
                    (memory_id,),
                )
                conn.commit()
                cursor.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not mark memory id=%d as enriched: %s", memory_id, exc)


def _enrich_single(llm_manager: "LLMManager", memory_id: int, content: str) -> bool:
    """Enrich a single memory entry with Wikipedia knowledge.

    Returns True if enrichment completed successfully (search terms obtained and
    Wikipedia was queried), False if it had to abort early due to an error.
    """
    from models import ChatMessage  # noqa: PLC0415
    from tools.wikipedia import wiki_search  # noqa: PLC0415

    # Step 1: Ask main LLM for relevant search terms
    prompt = _SEARCH_TERMS_PROMPT.format(
        max_terms=MAX_WIKI_QUERIES_PER_FACT,
        fact=content,
    )
    try:
        raw = llm_manager.chat_completion(
            model_name="main",
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=512,
        )
        # Strip <think>…</think> reasoning blocks emitted by some models.
        raw = re.sub(r"<think>.*?(?:</think>|$)", "", raw, flags=re.DOTALL).strip()
        # Some models return only a think-block with no actual output – treat as no terms.
        if not raw:
            search_terms: list[str] = []
        else:
            search_terms = json.loads(raw)
            if not isinstance(search_terms, list):
                search_terms = []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not get search terms for memory id=%d: %s", memory_id, exc)
        return False

    search_terms = [str(t) for t in search_terms if t][:MAX_WIKI_QUERIES_PER_FACT]
    if not search_terms:
        # LLM explicitly decided that no Wikipedia lookup is useful for this fact.
        # Mark as enriched=TRUE so we don't retry endlessly.
        logger.info(
            "Enrichment: LLM returned no search terms for memory id=%d – marking as enriched (no wiki needed).",
            memory_id,
        )
        return True

    logger.info("Enrichment: memory id=%d → search terms: %s", memory_id, search_terms)

    # Step 2: Search Wikipedia for each term and link results
    wiki_hits = 0    # total Wikipedia results found across all terms
    links_saved = 0  # how many were successfully written to the DB
    for term in search_terms:
        try:
            results = wiki_search(term, limit=MAX_WIKI_RESULTS_PER_TERM)
        except Exception as exc:  # noqa: BLE001
            logger.warning("wiki_search failed for term %r: %s", term, exc)
            continue

        logger.debug("Enrichment: term %r → %d Wikipedia result(s)", term, len(results))
        for result in results:
            wiki_hits += 1
            cache_id = _get_or_create_cache_id(result, memory_id)
            if cache_id is None:
                logger.warning(
                    "Enrichment: could not get/create cache entry for %r (memory id=%d) – DB write failed?",
                    result.get("title"), memory_id,
                )
                continue
            logger.debug("Enrichment: memory id=%d ↔ wiki_cache id=%d (%r)", memory_id, cache_id, result.get("title"))
            if _link_memory_to_cache(memory_id, cache_id):
                links_saved += 1
            _fill_cache_keywords(llm_manager, cache_id, result.get("summary", ""))

    # If Wikipedia returned results but nothing could be saved, the DB writes
    # all failed → report failure so the record stays unenriched and is retried.
    if wiki_hits > 0 and links_saved == 0:
        logger.warning(
            "Enrichment: %d Wikipedia hit(s) found but no links saved for memory id=%d – will retry.",
            wiki_hits, memory_id,
        )
        return False

    return True


def _get_or_create_cache_id(entry: dict, source_memory_id: int | None = None) -> int | None:
    """Return the wiki_cache.id for *entry*, inserting if needed.

    *source_memory_id* is stored on first insert (ignored on cache hit) so the
    entry can be traced back to the memory that triggered the enrichment.
    """
    try:
        from db.connection import get_connection  # noqa: PLC0415
        title = entry.get("title", "")
        lang = entry.get("lang", "de")
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM wiki_cache WHERE title = %s AND lang = %s LIMIT 1",
                (title, lang),
            )
            row = cursor.fetchone()
            if row:
                cursor.close()
                logger.debug("wiki_cache hit for %r (id=%d)", title, row[0])
                return row[0]
            cursor.execute(
                "INSERT INTO wiki_cache (title, query, summary, source_url, lang, fetched_at, source_memory_id) "
                "VALUES (%s, %s, %s, %s, %s, NOW(), %s)",
                (
                    title,
                    entry.get("query", ""),
                    entry.get("summary", ""),
                    entry.get("source_url", ""),
                    lang,
                    source_memory_id,
                ),
            )
            conn.commit()
            new_id = cursor.lastrowid
            cursor.close()
            logger.info("wiki_cache inserted %r as id=%d (source_memory_id=%s)", title, new_id, source_memory_id)
            return new_id
    except Exception as exc:  # noqa: BLE001
        logger.error("_get_or_create_cache_id error for %r: %s", entry.get("title"), exc)
        return None


def _link_memory_to_cache(memory_id: int, cache_id: int) -> bool:
    """Insert a memory_knowledge_link row (ignore duplicates).

    Returns True when a row was actually inserted (or already existed as a
    duplicate), False on DB error or when MariaDB silently swallowed an FK
    violation via INSERT IGNORE (rowcount == 0 but no exception).
    """
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT IGNORE INTO memory_knowledge_link (memory_id, cache_id) "
                "VALUES (%s, %s)",
                (memory_id, cache_id),
            )
            conn.commit()
            affected = cursor.rowcount  # 1 = inserted, 0 = duplicate or FK-violation (IGNORE)
            cursor.close()
        if affected == 0:
            # No FK constraints in this DB, so 0 rows = the link already exists (duplicate).
            # This is not an error – treat as success so partially-enriched memories can
            # be finalised on the next run instead of retrying forever.
            logger.debug(
                "_link_memory_to_cache: link memory_id=%d ↔ cache_id=%d already exists (duplicate, skipped)",
                memory_id, cache_id,
            )
            return True
        logger.info("memory_knowledge_link saved: memory_id=%d ↔ cache_id=%d", memory_id, cache_id)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("_link_memory_to_cache error (memory_id=%d, cache_id=%d): %s", memory_id, cache_id, exc)
        return False


def _fill_cache_keywords(llm_manager: "LLMManager", cache_id: int, summary: str) -> None:
    """Ask main LLM to extract keywords from *summary* and store them."""
    if not summary:
        return
    try:
        from models import ChatMessage  # noqa: PLC0415
        prompt = _KEYWORDS_PROMPT.format(summary=summary[:1000])
        keywords = llm_manager.chat_completion(
            model_name="main",
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=256,
        )
        # Strip <think>…</think> reasoning blocks emitted by some models.
        keywords = re.sub(r"<think>.*?(?:</think>|$)", "", keywords, flags=re.DOTALL).strip()
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE wiki_cache SET keywords = %s WHERE id = %s",
                (keywords, cache_id),
            )
            conn.commit()
            cursor.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("_fill_cache_keywords error for cache_id=%d: %s", cache_id, exc)


# ---------------------------------------------------------------------------
# Asyncio background loop
# ---------------------------------------------------------------------------

async def enrichment_loop(llm_manager: "LLMManager") -> None:
    """Async background task that runs enrichment every ENRICHMENT_INTERVAL_SECONDS.

    Register at startup with:
        asyncio.ensure_future(enrichment_loop(llm_manager))
    """
    logger.info(
        "Enrichment loop started (interval=%ds, max_per_run=%d).",
        ENRICHMENT_INTERVAL_SECONDS, MAX_ENRICHMENTS_PER_RUN,
    )
    while True:
        await asyncio.sleep(ENRICHMENT_INTERVAL_SECONDS)
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, enrich_pending_memories, llm_manager)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Enrichment loop iteration error: %s", exc)
