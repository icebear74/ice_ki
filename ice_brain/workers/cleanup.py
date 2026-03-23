"""
Cleanup Worker – consolidates, deduplicates, and cleans up memory entries.

What the cleanup worker does
-----------------------------
1. Delete expired temporal entries (expires_at < NOW()).
2. Find semantic duplicates (cosine similarity > 0.85) and merge them.
3. Resolve contradictions (high similarity but opposing content) – newer wins.
4. Merge related fragments into consolidated entries via LLM.

Configuration
-------------
CLEANUP_INTERVAL_SECONDS  – time between automatic cleanup runs (6 hours)
CLEANUP_SIMILARITY_THRESHOLD – cosine threshold for duplicate detection (0.85)
CLEANUP_MAX_PAIRS_PER_RUN   – max pairs to process per cleanup run
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

CLEANUP_INTERVAL_SECONDS: int = 6 * 60 * 60   # 6 hours
CLEANUP_SIMILARITY_THRESHOLD: float = 0.85
CLEANUP_MAX_PAIRS_PER_RUN: int = 10

_MERGE_PROMPT = """\
You are a memory consolidation assistant. Two memory entries are near-duplicates.
Merge them into ONE concise entry in German third person. Output ONLY the merged \
fact text – no JSON, no explanation.

Entry 1: {entry1}
Entry 2: {entry2}
"""

_CONTRADICTION_CHECK_PROMPT = """\
Do these two memory entries contradict each other? Answer with only "yes" or "no".

Entry 1: {entry1}
Entry 2: {entry2}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_cleanup_now(llm_manager: "LLMManager") -> dict:
    """Run all cleanup steps immediately.  Returns a summary dict."""
    summary: dict[str, int] = {
        "expired_deleted": 0,
        "duplicates_merged": 0,
        "contradictions_resolved": 0,
    }
    try:
        summary["expired_deleted"] = _cleanup_expired_all()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cleanup: _cleanup_expired_all failed: %s", exc)

    try:
        summary["duplicates_merged"], summary["contradictions_resolved"] = (
            _find_and_process_duplicates(llm_manager)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cleanup: duplicate processing failed: %s", exc)

    logger.info(
        "Cleanup run complete: expired=%d, merged=%d, contradictions=%d",
        summary["expired_deleted"], summary["duplicates_merged"],
        summary["contradictions_resolved"],
    )
    return summary


async def cleanup_loop(llm_manager: "LLMManager") -> None:
    """Background loop that runs cleanup every CLEANUP_INTERVAL_SECONDS."""
    logger.info(
        "Cleanup loop started (interval=%ds).", CLEANUP_INTERVAL_SECONDS
    )
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, run_cleanup_now, llm_manager)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cleanup loop iteration error: %s", exc)


# ---------------------------------------------------------------------------
# Step 1: Delete expired entries
# ---------------------------------------------------------------------------

def _cleanup_expired_all() -> int:
    """Delete all expired user_memory and relation_memory entries."""
    total = 0
    total += _cleanup_expired_table("user_memory")
    total += _cleanup_expired_table("relation_memory")
    return total


def _cleanup_expired_table(table: str) -> int:
    """Delete expired entries from a memory table. Returns count deleted."""
    # Whitelist to prevent SQL injection
    if table not in ("user_memory", "relation_memory"):
        return 0
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM `{table}` WHERE expires_at IS NOT NULL AND expires_at < NOW()"  # noqa: S608
            )
            count = cursor.rowcount
            conn.commit()
            cursor.close()
        if count > 0:
            logger.info("Cleanup: deleted %d expired entries from %s.", count, table)
        return count
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cleanup: _cleanup_expired_table(%s) error: %s", table, exc)
        return 0


# ---------------------------------------------------------------------------
# Step 2 & 3: Find and process duplicate pairs
# ---------------------------------------------------------------------------

def _find_duplicate_candidates(user_id: str) -> list[tuple[int, int, str, str, float]]:
    """Find pairs of user_memory entries with cosine similarity > threshold.

    Returns list of (id1, id2, content1, content2, score).
    """
    try:
        from db.connection import get_connection  # noqa: PLC0415
        from tools.embeddings import cosine_similarity, unpack_embedding  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cleanup: embeddings not available: %s", exc)
        return []

    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, content, embedding FROM user_memory "
                "WHERE user_id = %s AND embedding IS NOT NULL "
                "AND (expires_at IS NULL OR expires_at > NOW()) "
                "ORDER BY importance DESC LIMIT 200",
                (user_id,),
            )
            rows = cursor.fetchall()
            cursor.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cleanup: could not fetch user_memory rows: %s", exc)
        return []

    from tools.embeddings import cosine_similarity, unpack_embedding  # noqa: PLC0415

    pairs: list[tuple[int, int, str, str, float]] = []
    decoded = []
    for row_id, content, emb_raw in rows:
        try:
            vec = unpack_embedding(emb_raw)
            decoded.append((row_id, content, vec))
        except Exception:  # noqa: BLE001
            continue

    for i in range(len(decoded)):
        for j in range(i + 1, len(decoded)):
            id1, c1, v1 = decoded[i]
            id2, c2, v2 = decoded[j]
            score = cosine_similarity(v1, v2)
            if score >= CLEANUP_SIMILARITY_THRESHOLD:
                pairs.append((id1, id2, c1, c2, score))
            if len(pairs) >= CLEANUP_MAX_PAIRS_PER_RUN:
                return pairs
    return pairs


def _find_and_process_duplicates(llm_manager: "LLMManager") -> tuple[int, int]:
    """Process all users' memory duplicates.  Returns (merged, contradictions)."""
    if not llm_manager.is_ready("main"):
        logger.info("Cleanup: skipping duplicate processing – main model not loaded.")
        return 0, 0

    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT user_id FROM user_memory")
            user_ids = [r[0] for r in cursor.fetchall()]
            cursor.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cleanup: could not get user_ids: %s", exc)
        return 0, 0

    total_merged = 0
    total_contradictions = 0

    for user_id in user_ids:
        pairs = _find_duplicate_candidates(user_id)
        if not pairs:
            continue
        for id1, id2, c1, c2, score in pairs:
            try:
                merged, contra = _process_pair(llm_manager, id1, id2, c1, c2)
                total_merged += merged
                total_contradictions += contra
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cleanup: pair (%d,%d) processing failed: %s", id1, id2, exc)

    return total_merged, total_contradictions


def _process_pair(
    llm_manager: "LLMManager",
    id1: int, id2: int,
    content1: str, content2: str,
) -> tuple[int, int]:
    """Check if pair is duplicate or contradiction and process accordingly.

    Returns (merged:int, contradictions:int).
    """
    from models import ChatMessage  # noqa: PLC0415

    # Check for contradiction first
    contra_prompt = _CONTRADICTION_CHECK_PROMPT.format(entry1=content1, entry2=content2)
    try:
        contra_raw = llm_manager.chat_completion(
            model_name="main",
            messages=[ChatMessage(role="user", content=f"{contra_prompt}\n/no_think")],
            temperature=0.0,
            max_tokens=8,
        )
        contra_raw = re.sub(r"<think>.*?(?:</think>|$)", "", contra_raw, flags=re.DOTALL).strip().lower()
        is_contradiction = "yes" in contra_raw
    except Exception as exc:  # noqa: BLE001
        logger.debug("Cleanup: contradiction check failed: %s", exc)
        is_contradiction = False

    if is_contradiction:
        # Newer entry wins – delete the older one
        _delete_older_entry(id1, id2)
        logger.info("Cleanup: resolved contradiction between id=%d and id=%d.", id1, id2)
        return 0, 1

    # Merge duplicate
    merge_prompt = _MERGE_PROMPT.format(entry1=content1, entry2=content2)
    try:
        merged_content = llm_manager.chat_completion(
            model_name="main",
            messages=[ChatMessage(role="user", content=f"{merge_prompt}\n/no_think")],
            temperature=0.0,
            max_tokens=256,
        )
        merged_content = re.sub(
            r"<think>.*?(?:</think>|$)", "", merged_content, flags=re.DOTALL
        ).strip()
        if not merged_content:
            return 0, 0
    except Exception as exc:  # noqa: BLE001
        logger.debug("Cleanup: merge LLM call failed: %s", exc)
        return 0, 0

    _apply_merge(id1, id2, merged_content)
    logger.info("Cleanup: merged id=%d + id=%d → %r", id1, id2, merged_content[:80])
    return 1, 0


def _delete_older_entry(id1: int, id2: int) -> None:
    """Delete the older of two user_memory entries."""
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM user_memory WHERE id IN (%s, %s) ORDER BY created_at ASC LIMIT 1",
                (id1, id2),
            )
            row = cursor.fetchone()
            if row:
                cursor.execute("DELETE FROM user_memory WHERE id = %s", (row[0],))
                conn.commit()
            cursor.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cleanup: _delete_older_entry failed: %s", exc)


def _apply_merge(id1: int, id2: int, merged_content: str) -> None:
    """Replace two user_memory entries with one merged entry (keep id1, delete id2)."""
    try:
        from db.connection import get_connection  # noqa: PLC0415
        # Re-embed the merged content
        embedding_text: str | None = None
        try:
            from tools.embeddings import embed_one, vec_to_text  # noqa: PLC0415
            vec = embed_one(merged_content)
            embedding_text = vec_to_text(vec)
        except Exception:  # noqa: BLE001
            pass

        with get_connection() as conn:
            cursor = conn.cursor()
            if embedding_text:
                cursor.execute(
                    "UPDATE user_memory SET content = %s, embedding = VEC_FromText(%s), "
                    "updated_at = NOW() WHERE id = %s",
                    (merged_content, embedding_text, id1),
                )
            else:
                cursor.execute(
                    "UPDATE user_memory SET content = %s, updated_at = NOW() WHERE id = %s",
                    (merged_content, id1),
                )
            cursor.execute("DELETE FROM user_memory WHERE id = %s", (id2,))
            conn.commit()
            cursor.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cleanup: _apply_merge failed: %s", exc)
