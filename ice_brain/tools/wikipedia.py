"""
Wikipedia-Tool – fragt die deutsche Wikipedia REST API ab und cached
Ergebnisse in der MariaDB-Tabelle `wiki_cache`.

Öffentliche API
---------------
wiki_search(query, limit=3) -> list[dict]
    Sucht Artikel und gibt eine Liste von Zusammenfassungen zurück.

wiki_summary(title) -> dict | None
    Holt die Zusammenfassung eines einzelnen Artikels.

wiki_refresh(title) -> dict | None
    Invalidiert den Cache-Eintrag für *title* und lädt ihn neu.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_BASE_URL = "https://de.wikipedia.org"
_USER_AGENT = "ice_brain/1.0 (https://github.com/icebear74/ice_ki; contact via GitHub)"
_TIMEOUT = 5.0  # seconds

_UMLAUT_TABLE = str.maketrans({
    "ä": "ae", "ö": "oe", "ü": "ue",
    "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
    "ß": "ss",
})


def _transliterate_umlauts(text: str) -> str:
    """Replace German umlauts with their two-letter ASCII equivalents.

    The MediaWiki search API can fail to find articles when the search query
    contains umlauts (ä, ö, ü, ß), even on the German Wikipedia.  Transliterating
    them to ae/oe/ue/ss before searching improves recall significantly.
    Article *titles* returned by the API are kept as-is.
    """
    return text.translate(_UMLAUT_TABLE)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_client():
    """Return a configured httpx.Client instance."""
    try:
        import httpx  # noqa: PLC0415
        return httpx.Client(
            base_url=_BASE_URL,
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT,
        )
    except ImportError:
        raise RuntimeError("httpx is not installed. Run: pip install httpx")


# ---------------------------------------------------------------------------
# Wikipedia API calls (no cache)
# ---------------------------------------------------------------------------

def _api_summary(title: str) -> dict | None:
    """Fetch the REST summary for a single article title."""
    try:
        with _get_client() as client:
            resp = client.get(f"/api/rest_v1/page/summary/{title}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Wikipedia API error for title %r: %s", title, exc)
        return None


def _api_full_text(title: str) -> str:
    """Fetch the full plain-text content of a Wikipedia article.

    Uses the MediaWiki action=query&prop=extracts API with explaintext=true
    to retrieve the article as clean plain text without markup.
    Returns an empty string on error or when the article is not found.
    """
    try:
        with _get_client() as client:
            resp = client.get(
                "/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": True,
                    "titles": title,
                    "format": "json",
                    "utf8": 1,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                return page.get("extract", "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Wikipedia full-text error for title %r: %s", title, exc)
    return ""


def _api_search(query: str, limit: int = 3) -> list[dict]:
    """Search Wikipedia and return up to *limit* summaries with full text."""
    try:
        with _get_client() as client:
            resp = client.get(
                "/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": _transliterate_umlauts(query),
                    "srlimit": limit,
                    "format": "json",
                    "utf8": 1,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Wikipedia search error for query %r: %s", query, exc)
        return []

    results = []
    for hit in data.get("query", {}).get("search", []):
        title = hit.get("title", "")
        summary_data = _api_summary(title)
        if summary_data:
            full_text = _api_full_text(title)
            results.append(_normalise_summary(summary_data, query, full_text))
    return results


def _normalise_summary(data: dict, query: str = "", full_text: str = "") -> dict:
    """Extract relevant fields from a Wikipedia REST summary response."""
    result = {
        "title": data.get("title", ""),
        "query": query,
        "summary": data.get("extract", ""),
        "full_text": full_text,
        "source_url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        "lang": "de",
    }
    # Extract thumbnail image URL when available
    thumbnail = data.get("thumbnail") or data.get("originalimage")
    if isinstance(thumbnail, dict):
        image_url = thumbnail.get("source", "")
        if image_url:
            result["image_url"] = image_url
    return result


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_get(title: str, lang: str = "de") -> dict | None:
    """Return a cached wiki entry if it exists and hasn't expired, else None."""
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, title, query, summary, full_text, source_url, lang, fetched_at, ttl_days "
                "FROM wiki_cache "
                "WHERE title = %s AND lang = %s "
                "AND DATE_ADD(fetched_at, INTERVAL ttl_days DAY) > NOW() "
                "LIMIT 1",
                (title, lang),
            )
            row = cursor.fetchone()
            cursor.close()
        if not row:
            return None
        return {
            "id": row[0],
            "title": row[1],
            "query": row[2],
            "summary": row[3],
            "full_text": row[4] or "",
            "source_url": row[5],
            "lang": row[6],
            "fetched_at": str(row[7]),
            "ttl_days": row[8],
            "image_url": row[9] or None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("wiki_cache read error: %s", exc)
        return None


def _cache_set(entry: dict) -> None:
    """Insert or update a wiki_cache row."""
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO wiki_cache (title, query, summary, full_text, source_url, lang, image_url, fetched_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, NOW()) "
                "ON DUPLICATE KEY UPDATE "
                "query = VALUES(query), "
                "summary = VALUES(summary), "
                "full_text = VALUES(full_text), "
                "source_url = VALUES(source_url), "
                "image_url = COALESCE(VALUES(image_url), image_url), "
                "fetched_at = NOW()",
                (
                    entry.get("title", ""),
                    entry.get("query", ""),
                    entry.get("summary", ""),
                    entry.get("full_text", "") or "",
                    entry.get("source_url", ""),
                    entry.get("lang", "de"),
                    entry.get("image_url") or None,
                ),
            )
            conn.commit()
            cursor.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("wiki_cache write error: %s", exc)


def _cache_delete(title: str, lang: str = "de") -> None:
    """Delete a wiki_cache entry by title + lang."""
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM wiki_cache WHERE title = %s AND lang = %s",
                (title, lang),
            )
            conn.commit()
            cursor.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("wiki_cache delete error for title %r: %s", title, exc)


def wiki_topic_is_stale(topic: str, lang: str = "de", max_age_days: int = 7) -> bool:
    """Return True when the wiki_cache has no fresh entry for *topic*.

    A "fresh" entry is one whose ``fetched_at`` is within *max_age_days* days.
    If no entry exists at all, or all matching entries are older than
    *max_age_days*, True (= stale) is returned so the caller knows to perform a
    live lookup.

    The match is intentionally broad: the *topic* keywords are searched inside
    both the ``title`` and the ``query`` columns so that short extracted topics
    (e.g. "Schindler Martin") still find articles cached under their full title.

    On any DB error the function returns True (= assume stale, better to
    over-fetch than to serve outdated data).
    """
    if not topic or not topic.strip():
        return True
    try:
        from db.connection import get_connection  # noqa: PLC0415
        # Build a LIKE pattern from each word in the topic so that "Martin
        # Schindler" will match both "Martin Schindler" and "Schindler, Martin".
        words = [w for w in topic.split() if len(w) > 1]
        if not words:
            return True
        with get_connection() as conn:
            cursor = conn.cursor()
            # Check whether ANY matching row was fetched within max_age_days.
            # A single fresh row is sufficient to consider the topic cached.
            like_clauses = " OR ".join(
                "(title LIKE %s OR query LIKE %s)" for _ in words
            )
            params: list[str | int] = []
            for w in words:
                pattern = f"%{w}%"
                params.extend([pattern, pattern])
            params.append(max_age_days)
            cursor.execute(
                f"SELECT 1 FROM wiki_cache "  # noqa: S608
                f"WHERE lang = %s AND ({like_clauses}) "
                f"AND fetched_at >= DATE_SUB(NOW(), INTERVAL %s DAY) "
                f"LIMIT 1",
                [lang, *params],
            )
            row = cursor.fetchone()
            cursor.close()
        is_stale = row is None
        logger.debug(
            "wiki_topic_is_stale(%r, max_age_days=%d): %s",
            topic, max_age_days, "stale" if is_stale else "fresh",
        )
        return is_stale
    except Exception as exc:  # noqa: BLE001
        logger.warning("wiki_topic_is_stale check failed (assuming stale): %s", exc)
        return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def wiki_summary(title: str) -> dict | None:
    """Return the summary for a Wikipedia article, using the cache.

    Returns a dict with keys: title, query, summary, full_text, source_url, lang.
    Returns None if the article does not exist.
    """
    cached = _cache_get(title)
    if cached:
        logger.debug("wiki_summary cache hit: %r", title)
        return cached

    data = _api_summary(title)
    if data is None:
        return None

    full_text = _api_full_text(title)
    entry = _normalise_summary(data, full_text=full_text)
    _cache_set(entry)
    return entry


def wiki_search(query: str, limit: int = 3) -> list[dict]:
    """Search Wikipedia for *query* and return up to *limit* article summaries.

    Each result is a dict with keys: title, query, summary, source_url, lang.
    Results are cached per article title.
    """
    results = _api_search(query, limit=limit)
    for entry in results:
        _cache_set(entry)
    return results


def wiki_live_lookup(query: str, limit: int = 2) -> list[dict]:
    """Fetch fresh Wikipedia results for *query*, bypassing the local cache.

    Unlike :func:`wiki_search`, this function:
    * always calls the Wikipedia API (ignores cached TTL),
    * force-refreshes the ``wiki_cache`` row for each result,
    * deletes and re-stores the ``wiki_chunks`` so the vector search index
      reflects the latest article content.

    Use this when the user has signalled that the AI gave incorrect information
    and needs up-to-date facts.

    Returns a list of result dicts (same shape as :func:`wiki_search`).
    """
    results = _api_search(query, limit=limit)
    for entry in results:
        title = entry.get("title", "")
        lang = entry.get("lang", "de")

        # Force-refresh the cache row
        _cache_delete(title, lang)
        _cache_set(entry)

        # Get the id of the just-written cache row
        cache_id: int | None = None
        try:
            from db.connection import get_connection  # noqa: PLC0415
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM wiki_cache WHERE title = %s AND lang = %s LIMIT 1",
                    (title, lang),
                )
                row = cursor.fetchone()
                cursor.close()
            cache_id = row[0] if row else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("wiki_live_lookup: could not fetch cache id for %r: %s", title, exc)

        if cache_id is not None:
            try:
                from db.wiki import refresh_article_chunks  # noqa: PLC0415
                refresh_article_chunks(
                    cache_id,
                    title,
                    entry.get("full_text") or entry.get("summary", ""),
                    lang,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("wiki_live_lookup: chunk refresh failed for %r: %s", title, exc)

    logger.info("wiki_live_lookup: refreshed %d article(s) for query %r.", len(results), query)
    return results


def wiki_refresh(title: str) -> dict | None:
    """Invalidate the cache entry for *title* and re-fetch from Wikipedia.

    Returns the fresh entry, or None if the article does not exist.
    """
    _cache_delete(title)
    return wiki_summary(title)
