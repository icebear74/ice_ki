"""
wiki_chunks – Chunking, Embedding und Vektorsuche für Wikipedia-Artikel.

Wikipedia-Artikel werden in Textabschnitte (Chunks) aufgeteilt, embeddiert
und in der Tabelle `wiki_chunks` gespeichert.  Beim Chatten sucht
search_wiki_chunks() über Python-seitige Cosine-Similarity die relevantesten
Passagen heraus und gibt sie zurück, damit der LLM sie in seinen Kontext
einbeziehen kann.

Öffentliche API
---------------
store_article_chunks(wiki_cache_id, title, full_text, lang)
    Zerlegt den Artikel in Chunks, bettet sie ein und schreibt sie in DB.
    Idempotent – läuft leer wenn Chunks für diesen Artikel schon existieren.

search_wiki_chunks(query, limit=5) -> list[dict]
    Embed die Anfrage, berechnet Cosine-Similarity gegen alle gespeicherten
    Chunks und gibt die Top-K zurück (keys: title, content, chunk_idx,
    article_id, score).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500      # Zeichen pro Chunk
CHUNK_OVERLAP = 100   # Überlappung zwischen benachbarten Chunks
MIN_CHUNK_CHARS = 80  # Chunks kürzer als dieser Wert werden verworfen
MAX_VECTOR_SEARCH_ROWS = 20_000  # Safety cap für den In-Memory-Cosine-Scan


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_chunks(text: str) -> list[str]:
    """Split *text* into overlapping character-window chunks.

    Returns a list of non-empty strings, each at least MIN_CHUNK_CHARS long.
    """
    if not text or not text.strip():
        return []
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        if end >= length:
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def store_article_chunks(
    wiki_cache_id: int,
    title: str,
    full_text: str,
    lang: str = "de",
) -> int:
    """Chunk, embed and persist a Wikipedia article in `wiki_chunks`.

    Returns the number of newly stored chunks.
    Idempotent: returns 0 immediately when chunks already exist for this article.
    Never raises – all errors are logged.
    """
    if not full_text or not full_text.strip():
        logger.debug("store_article_chunks: no text for wiki_cache_id=%d – skipped.", wiki_cache_id)
        return 0

    # Idempotency check
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM wiki_chunks WHERE article_id = %s",
                (wiki_cache_id,),
            )
            count = cursor.fetchone()[0]
            cursor.close()
        if count > 0:
            logger.debug(
                "store_article_chunks: %d chunk(s) already exist for wiki_cache_id=%d – skipped.",
                count, wiki_cache_id,
            )
            return 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("store_article_chunks: idempotency check failed (wiki_cache_id=%d): %s", wiki_cache_id, exc)
        return 0

    chunks = _split_chunks(full_text)
    if not chunks:
        logger.debug("store_article_chunks: text for wiki_cache_id=%d produced no chunks.", wiki_cache_id)
        return 0

    # Embed all chunks in one batch
    try:
        from tools.embeddings import embed, vec_to_text  # noqa: PLC0415
        vectors = embed(chunks)
    except Exception as exc:  # noqa: BLE001
        logger.warning("store_article_chunks: embedding failed for wiki_cache_id=%d: %s", wiki_cache_id, exc)
        return 0

    stored = 0
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
                try:
                    cursor.execute(
                        # VEC_FromText() parses the JSON-array string '[f1,f2,…]'.
                        # Passing raw bytes via %s causes charset-encoding corruption
                        # (mysql-connector applies utf8mb4 encoding to bytes params).
                        "INSERT INTO wiki_chunks "
                        "(article_id, title, chunk_idx, content, lang, embedding) "
                        "VALUES (%s, %s, %s, %s, %s, VEC_FromText(%s))",
                        (wiki_cache_id, title, idx, chunk, lang, vec_to_text(vec)),
                    )
                    stored += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "store_article_chunks: could not store chunk idx=%d "
                        "for wiki_cache_id=%d: %s",
                        idx, wiki_cache_id, exc,
                    )
            conn.commit()
            cursor.close()
        logger.info(
            "store_article_chunks: %d chunk(s) stored for %r (wiki_cache_id=%d).",
            stored, title, wiki_cache_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("store_article_chunks: DB error for wiki_cache_id=%d: %s", wiki_cache_id, exc)

    return stored


def refresh_article_chunks(
    wiki_cache_id: int,
    title: str,
    full_text: str,
    lang: str = "de",
) -> int:
    """Delete all existing chunks for *wiki_cache_id* then re-store fresh ones.

    Unlike :func:`store_article_chunks` (which is idempotent and skips when
    chunks already exist), this function always replaces the stored content.
    Returns the number of newly stored chunks.
    """
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM wiki_chunks WHERE article_id = %s",
                (wiki_cache_id,),
            )
            conn.commit()
            cursor.close()
        logger.debug("refresh_article_chunks: deleted old chunks for wiki_cache_id=%d.", wiki_cache_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "refresh_article_chunks: could not delete old chunks for wiki_cache_id=%d: %s",
            wiki_cache_id, exc,
        )
        return 0

    return store_article_chunks(wiki_cache_id, title, full_text, lang)


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def search_wiki_chunks(query: str, limit: int = 5) -> list[dict]:
    """Find the most relevant wiki chunks for *query* via cosine similarity.

    Embeds *query*, fetches all stored chunk embeddings from `wiki_chunks`,
    computes cosine similarity in Python and returns the top *limit* results.

    Each result dict has keys:
        id, article_id, title, chunk_idx, content, score (float 0-1)

    Returns [] on error or when no chunks are stored yet.
    """
    if not query or not query.strip():
        return []

    # Embed the query
    try:
        from tools.embeddings import embed_one, cosine_similarity, unpack_embedding  # noqa: PLC0415
        query_vec = embed_one(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("search_wiki_chunks: could not embed query: %s", exc)
        return []

    # Fetch all chunk embeddings (safety cap: MAX_VECTOR_SEARCH_ROWS)
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT wc.id, wc.article_id, wc.title, wc.chunk_idx, wc.content, "
                "       wc.embedding, wcache.image_url "
                "FROM wiki_chunks wc "
                "LEFT JOIN wiki_cache wcache ON wcache.id = wc.article_id "
                "WHERE wc.embedding IS NOT NULL "
                "LIMIT %s",
                (MAX_VECTOR_SEARCH_ROWS,),
            )
            rows = cursor.fetchall()
            cursor.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("search_wiki_chunks: DB fetch failed: %s", exc)
        return []

    if not rows:
        return []

    scored: list[dict] = []
    for row_id, article_id, title, chunk_idx, content, embedding_bytes, image_url in rows:
        try:
            vec = unpack_embedding(embedding_bytes)
            score = cosine_similarity(query_vec, vec)
            scored.append({
                "id": row_id,
                "article_id": article_id,
                "title": title,
                "chunk_idx": chunk_idx,
                "content": content,
                "score": score,
                "image_url": image_url or None,
            })
        except Exception:  # noqa: BLE001
            continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


def search_wiki_chunks_by_title(query: str, limit: int = 3) -> list[dict]:
    """Find wiki chunks whose article title contains words from *query*.

    This is a fast SQL-side fallback used when vector (cosine) search misses
    the target because the embedding is dominated by meta-question semantics
    (e.g. "was weißt du über martin schindler" → embedding drifts away from
    the entity).  A direct title match reliably surfaces the correct article
    (e.g. "Martin Schindler") regardless of embedding quality.

    Returns the first *limit* chunks for matching articles, ordered by
    chunk_idx (i.e. the start of the article comes first).  Each result dict
    has the same shape as :func:`search_wiki_chunks` with ``score=1.0``.
    Returns [] on error or when no words can be extracted from *query*.
    """
    import re  # noqa: PLC0415

    # Extract words ≥ 3 chars to build LIKE clauses.  We intentionally keep
    # all words (including lowercase) so that "martin schindler" works even
    # though neither word is stop-word filtered here.
    words = [w for w in re.findall(r"[A-Za-zÄÖÜäöüß]{3,}", query)]
    if not words:
        return []

    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            # Require ALL words to appear in the title (AND logic) so that
            # "martin schindler" only matches "Martin Schindler", not just
            # any article with "Martin" or any article with "Schindler".
            like_clauses = " AND ".join("wc.title LIKE %s" for _ in words)
            params = [f"%{w}%" for w in words]
            cursor.execute(
                f"SELECT wc.id, wc.article_id, wc.title, wc.chunk_idx, "  # noqa: S608
                f"       wc.content, wcache.image_url "
                f"FROM wiki_chunks wc "
                f"LEFT JOIN wiki_cache wcache ON wcache.id = wc.article_id "
                f"WHERE {like_clauses} "
                f"ORDER BY wc.article_id, wc.chunk_idx ASC "
                f"LIMIT %s",
                [*params, limit],
            )
            rows = cursor.fetchall()
            cursor.close()
        return [
            {
                "id": row[0],
                "article_id": row[1],
                "title": row[2],
                "chunk_idx": row[3],
                "content": row[4],
                "score": 1.0,
                "image_url": row[5] or None,
            }
            for row in rows
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("search_wiki_chunks_by_title error for query %r: %s", query, exc)
        return []
