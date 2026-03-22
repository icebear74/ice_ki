"""
Dummy weather tool – example implementation that shows the tool pattern.

In Phase 5+ this will be replaced (or complemented) by a real weather API call.
Register it so the registry is non-empty from the start.
"""

from __future__ import annotations

from . import register_tool


@register_tool("weather")
def get_weather(location: str = "unknown") -> dict:
    """Return fake weather data for *location*.

    Returns a dict that the orchestrator can format into a context snippet.
    """
    return {
        "location": location,
        "temperature_c": 20,
        "condition": "sonnig (Dummy-Daten)",
        "humidity_pct": 55,
        "note": "Dies sind Platzhalterdaten. Echte API-Anbindung folgt in Phase 5.",
    }
