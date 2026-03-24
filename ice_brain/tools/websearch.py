"""
DuckDuckGo web search and news search tools for ice_brain.

Uses the ``duckduckgo-search`` package (no API key required).

Public API
----------
web_search(query, max_results=5) -> list[dict]
    Performs a web search and returns title, url, snippet for each result.

news_search(query, max_results=5, timelimit=None) -> list[dict]
    Searches news results with an optional time filter
    (``d`` = day, ``w`` = week, ``m`` = month).
"""

from __future__ import annotations

import logging

from tools import register_tool

logger = logging.getLogger(__name__)


@register_tool("web_search")
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using DuckDuckGo and return structured results.

    Each result dict contains ``title``, ``url``, and ``snippet``.
    Returns an empty list on error.
    """
    try:
        from duckduckgo_search import DDGS  # noqa: PLC0415
        results = DDGS().text(query, max_results=max_results, region="de-de")
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in (results or [])
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("web_search error for query %r: %s", query, exc)
        return []


@register_tool("news_search")
def news_search(
    query: str,
    max_results: int = 5,
    timelimit: str | None = None,
) -> list[dict]:
    """Search recent news using DuckDuckGo.

    *timelimit* controls how fresh the results must be:
      ``"d"`` = last day, ``"w"`` = last week, ``"m"`` = last month.
    Each result dict contains ``title``, ``url``, ``snippet``, ``date``,
    and ``source``.
    Returns an empty list on error.
    """
    try:
        from duckduckgo_search import DDGS  # noqa: PLC0415
        kwargs: dict = {"max_results": max_results, "region": "de-de"}
        if timelimit:
            kwargs["timelimit"] = timelimit
        results = DDGS().news(query, **kwargs)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("body", ""),
                "date": r.get("date", ""),
                "source": r.get("source", ""),
            }
            for r in (results or [])
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("news_search error for query %r: %s", query, exc)
        return []
