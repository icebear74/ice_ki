"""
Relations – CRUD operations for the relations + relation_memory tables.

A "relation" is a person the user knows (friend, partner, family member, etc.).
Each relation can have its own set of memories (facts) stored in relation_memory.

Public API
----------
find_relation(user_id, name)
    Return the relation id for a named person, or None if not found.

find_or_create_relation(user_id, name, relation_type, relation_detail=None)
    Return (relation_id, created:bool).  Creates the relation if not present.

get_relation(relation_id)
    Return the full relation row as a dict, or None.

get_relations_for_user(user_id)
    Return a list of relation dicts for a user.

upsert_relation_fact(relation_id, fact)
    Insert or update a single fact in relation_memory (with embedding).

get_relation_facts(relation_id, category=None, active_only=True)
    Return a list of fact dicts for a relation.

semantic_search_relation_memory(user_id, query_vec, limit=5)
    Cross-relation cosine similarity search; returns top-limit results.

set_relation_confirmed(relation_id, confirmed=True, relation_type=None)
    Mark a relation as confirmed (disambiguation resolved).

get_pending_disambiguation(user_id)
    Return first unconfirmed relation for user, or None.
"""

from __future__ import annotations

import difflib
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_VALID_RELATION_TYPES = {
    "partner", "friend", "family", "colleague", "acquaintance", "other",
}

# Same categories as user_memory
_VALID_CATEGORIES = {
    "preference", "personal", "relationship", "hobby", "experience",
    "activity", "mood", "plan", "topic",
}

_TTL_MAP: dict[str, int] = {
    "1h": 1, "2h": 2, "4h": 4, "8h": 8, "24h": 24, "48h": 48,
}

_MAX_SIMILARITY_SEARCH_ROWS = 50
_MIN_WORD_LENGTH_FOR_SIMILARITY = 4
_SIMILARITY_STOP_WORDS: frozenset[str] = frozenset({
    "möchte", "mag", "will", "muss", "kann", "soll",
    "hat", "hatte", "ist", "war", "sind", "waren",
    "likes", "loves", "hates", "wants", "has", "is", "was",
})


# ---------------------------------------------------------------------------
# Relation CRUD
# ---------------------------------------------------------------------------

def find_relation(user_id: str, name: str) -> int | None:
    """Return the relation id for *name* (case-insensitive), or None."""
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM relations WHERE user_id = %s AND LOWER(name) = LOWER(%s) LIMIT 1",
                (user_id, name),
            )
            row = cursor.fetchone()
            cursor.close()
        return row[0] if row else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("find_relation error: %s", exc)
        return None


def find_or_create_relation(
    user_id: str,
    name: str,
    relation_type: str,
    relation_detail: str | None = None,
) -> tuple[int, bool]:
    """Return (relation_id, created).

    If a relation with the same name (case-insensitive) exists, return it.
    Otherwise create a new one (unconfirmed by default).
    """
    if relation_type not in _VALID_RELATION_TYPES:
        relation_type = "other"
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM relations WHERE user_id = %s AND LOWER(name) = LOWER(%s) LIMIT 1",
                (user_id, name),
            )
            row = cursor.fetchone()
            if row:
                cursor.close()
                return row[0], False
            cursor.execute(
                "INSERT INTO relations (user_id, name, relation_type, relation_detail, confirmed) "
                "VALUES (%s, %s, %s, %s, FALSE)",
                (user_id, name.strip(), relation_type, relation_detail),
            )
            conn.commit()
            new_id = cursor.lastrowid
            cursor.close()
        logger.info("Created relation %r (id=%d, type=%s) for user %r.", name, new_id, relation_type, user_id)
        return new_id, True
    except Exception as exc:  # noqa: BLE001
        logger.error("find_or_create_relation error: %s", exc)
        raise


def get_relation(relation_id: int) -> dict | None:
    """Return full relation row as dict, or None."""
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, user_id, name, relation_type, relation_detail, confirmed, enrichment_done, "
                "created_at, updated_at FROM relations WHERE id = %s",
                (relation_id,),
            )
            row = cursor.fetchone()
            cursor.close()
        if row is None:
            return None
        return {
            "id": row[0], "user_id": row[1], "name": row[2],
            "relation_type": row[3], "relation_detail": row[4],
            "confirmed": bool(row[5]), "enrichment_done": bool(row[6]),
            "created_at": row[7], "updated_at": row[8],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_relation error: %s", exc)
        return None


def get_relations_for_user(user_id: str) -> list[dict]:
    """Return all relations for *user_id*."""
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, relation_type, relation_detail, confirmed, enrichment_done "
                "FROM relations WHERE user_id = %s ORDER BY name",
                (user_id,),
            )
            rows = cursor.fetchall()
            cursor.close()
        return [
            {
                "id": r[0], "name": r[1], "relation_type": r[2],
                "relation_detail": r[3], "confirmed": bool(r[4]),
                "enrichment_done": bool(r[5]),
            }
            for r in rows
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_relations_for_user error: %s", exc)
        return []


def set_relation_confirmed(
    relation_id: int,
    confirmed: bool = True,
    relation_type: str | None = None,
) -> None:
    """Mark a relation as confirmed and optionally update its type."""
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            if relation_type and relation_type in _VALID_RELATION_TYPES:
                cursor.execute(
                    "UPDATE relations SET confirmed = %s, relation_type = %s WHERE id = %s",
                    (confirmed, relation_type, relation_id),
                )
            else:
                cursor.execute(
                    "UPDATE relations SET confirmed = %s WHERE id = %s",
                    (confirmed, relation_id),
                )
            conn.commit()
            cursor.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("set_relation_confirmed error: %s", exc)


def get_pending_disambiguation(user_id: str) -> dict | None:
    """Return the first unconfirmed relation for user, or None."""
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, relation_type FROM relations "
                "WHERE user_id = %s AND confirmed = FALSE ORDER BY created_at LIMIT 1",
                (user_id,),
            )
            row = cursor.fetchone()
            cursor.close()
        if row is None:
            return None
        return {"id": row[0], "name": row[1], "relation_type": row[2]}
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_pending_disambiguation error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Relation Memory CRUD
# ---------------------------------------------------------------------------

def _compute_expires_at(ttl_str: str | None) -> datetime | None:
    if ttl_str is None:
        return None
    hours = _TTL_MAP.get(str(ttl_str))
    if hours is None:
        return None
    return datetime.now(tz=timezone.utc) + timedelta(hours=hours)


def _find_similar_relation_fact(cursor, relation_id: int, category: str, content: str) -> int | None:
    """Return id of an existing similar relation_memory row, or None."""
    cursor.execute(
        "SELECT id, content FROM relation_memory "
        "WHERE relation_id = %s AND category = %s "
        "AND (expires_at IS NULL OR expires_at > NOW()) "
        "LIMIT %s",
        (relation_id, category, _MAX_SIMILARITY_SEARCH_ROWS),
    )
    rows = cursor.fetchall()
    if not rows:
        return None

    words = [
        w.lower()
        for w in content.split()
        if len(w) >= _MIN_WORD_LENGTH_FOR_SIMILARITY
        and w.lower().rstrip(",.;:!?") not in _SIMILARITY_STOP_WORDS
    ]
    content_lower = content.lower()

    for row_id, row_content in rows:
        row_lower = row_content.lower()
        if words:
            for word in words[:2]:
                if word in row_lower:
                    return row_id
        ratio = difflib.SequenceMatcher(None, content_lower, row_lower).ratio()
        if ratio >= 0.75:
            return row_id
    return None


def upsert_relation_fact(relation_id: int, fact: dict) -> None:
    """Insert or update a single fact in relation_memory (with embedding).

    *fact* must have keys: content, category, importance, ttl, temporal.
    """
    content = str(fact.get("content", "")).strip()
    if not content:
        return

    category = str(fact.get("category", "preference")).lower()
    if category not in _VALID_CATEGORIES:
        category = "preference"

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

    expires_at = _compute_expires_at(ttl_str)

    # Compute embedding
    embedding_text: str | None = None
    try:
        from tools.embeddings import embed_one, vec_to_text  # noqa: PLC0415
        vec = embed_one(content)
        embedding_text = vec_to_text(vec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("upsert_relation_fact: embedding failed: %s", exc)

    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            existing_id = _find_similar_relation_fact(cursor, relation_id, category, content)
            if existing_id is not None:
                if embedding_text:
                    cursor.execute(
                        "UPDATE relation_memory SET content = %s, importance = %s, "
                        "temporal = %s, updated_at = NOW(), expires_at = %s, "
                        "embedding = VEC_FromText(%s) "
                        "WHERE id = %s",
                        (content, importance, temporal, expires_at, embedding_text, existing_id),
                    )
                else:
                    cursor.execute(
                        "UPDATE relation_memory SET content = %s, importance = %s, "
                        "temporal = %s, updated_at = NOW(), expires_at = %s "
                        "WHERE id = %s",
                        (content, importance, temporal, expires_at, existing_id),
                    )
                logger.debug("relation_memory updated (id=%d): %s", existing_id, content)
            else:
                if embedding_text:
                    cursor.execute(
                        "INSERT INTO relation_memory "
                        "(relation_id, category, content, importance, temporal, expires_at, embedding) "
                        "VALUES (%s, %s, %s, %s, %s, %s, VEC_FromText(%s))",
                        (relation_id, category, content, importance, temporal, expires_at, embedding_text),
                    )
                else:
                    cursor.execute(
                        "INSERT INTO relation_memory "
                        "(relation_id, category, content, importance, temporal, expires_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (relation_id, category, content, importance, temporal, expires_at),
                    )
                logger.debug("relation_memory inserted (relation_id=%d): %s", relation_id, content)
            conn.commit()
            cursor.close()
    except Exception as exc:  # noqa: BLE001
        logger.error("upsert_relation_fact error: %s", exc)
        raise


def get_relation_facts(
    relation_id: int,
    category: str | None = None,
    active_only: bool = True,
) -> list[dict]:
    """Return a list of fact dicts for *relation_id*."""
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            if category:
                if active_only:
                    cursor.execute(
                        "SELECT id, category, content, importance, temporal, expires_at "
                        "FROM relation_memory WHERE relation_id = %s AND category = %s "
                        "AND (expires_at IS NULL OR expires_at > NOW()) "
                        "ORDER BY importance DESC",
                        (relation_id, category),
                    )
                else:
                    cursor.execute(
                        "SELECT id, category, content, importance, temporal, expires_at "
                        "FROM relation_memory WHERE relation_id = %s AND category = %s "
                        "ORDER BY importance DESC",
                        (relation_id, category),
                    )
            else:
                if active_only:
                    cursor.execute(
                        "SELECT id, category, content, importance, temporal, expires_at "
                        "FROM relation_memory WHERE relation_id = %s "
                        "AND (expires_at IS NULL OR expires_at > NOW()) "
                        "ORDER BY importance DESC",
                        (relation_id,),
                    )
                else:
                    cursor.execute(
                        "SELECT id, category, content, importance, temporal, expires_at "
                        "FROM relation_memory WHERE relation_id = %s ORDER BY importance DESC",
                        (relation_id,),
                    )
            rows = cursor.fetchall()
            cursor.close()
        return [
            {
                "id": r[0], "category": r[1], "content": r[2],
                "importance": r[3], "temporal": r[4], "expires_at": r[5],
            }
            for r in rows
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_relation_facts error: %s", exc)
        return []


def get_all_relation_facts_for_user(user_id: str) -> list[dict]:
    """Return all active relation facts for user, with relation info attached."""
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT rm.id, rm.relation_id, r.name, r.relation_type, "
                "rm.category, rm.content, rm.importance, rm.temporal, rm.embedding "
                "FROM relation_memory rm "
                "JOIN relations r ON r.id = rm.relation_id "
                "WHERE r.user_id = %s "
                "AND (rm.expires_at IS NULL OR rm.expires_at > NOW()) "
                "ORDER BY rm.importance DESC",
                (user_id,),
            )
            rows = cursor.fetchall()
            cursor.close()
        return [
            {
                "id": r[0], "relation_id": r[1], "name": r[2],
                "relation_type": r[3], "category": r[4], "content": r[5],
                "importance": r[6], "temporal": r[7], "embedding": r[8],
            }
            for r in rows
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_all_relation_facts_for_user error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Semantic search over relation_memory
# ---------------------------------------------------------------------------

def semantic_search_relation_memory(
    user_id: str,
    query_vec: np.ndarray,
    limit: int = 5,
) -> list[dict]:
    """Cosine similarity search over all relation_memory entries for *user_id*.

    Returns up to *limit* results sorted by score descending.
    Each result: {id, relation_id, name, relation_type, category, content,
                  importance, temporal, score}
    """
    rows = get_all_relation_facts_for_user(user_id)
    if not rows:
        return []

    try:
        from tools.embeddings import cosine_similarity, unpack_embedding  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        logger.warning("semantic_search_relation_memory: embeddings not available: %s", exc)
        return []

    scored: list[tuple[float, dict]] = []
    for row in rows:
        raw = row.get("embedding")
        if raw is None:
            continue
        try:
            vec = unpack_embedding(raw)
            score = cosine_similarity(query_vec, vec)
            scored.append((score, row))
        except Exception:  # noqa: BLE001
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, row in scored[:limit]:
        r = dict(row)
        r["score"] = score
        r.pop("embedding", None)
        results.append(r)
    return results
