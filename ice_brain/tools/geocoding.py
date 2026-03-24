"""
Geocoding – Ortsname zu Koordinaten auflösen via OpenStreetMap/Nominatim.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# In-memory-Cache: Ortsname (lowercase) → dict mit lat/lon/display_name
_cache: dict[str, dict[str, Any]] = {}
_cache_lock = threading.Lock()

# Nominatim rate-limiting: max 1 Anfrage pro Sekunde
_last_request_time: float = 0.0
_rate_lock = threading.Lock()

_USER_AGENT = "ice_brain/1.0 (https://github.com/icebear74/ice_ki)"
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def geocode(place_name: str) -> dict | None:
    """Löst einen Ortsnamen in Koordinaten auf (OpenStreetMap/Nominatim).

    Gibt ein dict zurück: {"lat": float, "lon": float, "display_name": str}
    oder None bei Fehler / unbekanntem Ort.
    """
    if not place_name or not place_name.strip():
        return None

    key = place_name.strip().lower()

    # Aus Cache zurückgeben wenn vorhanden
    with _cache_lock:
        if key in _cache:
            logger.debug("Geocoding cache hit for %r", place_name)
            return _cache[key]

    # Rate-Limiting: maximal 1 Request/Sekunde (Nominatim Usage Policy)
    with _rate_lock:
        global _last_request_time  # noqa: PLW0603
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        _last_request_time = time.monotonic()

    try:
        import httpx  # noqa: PLC0415
        params = {
            "q": place_name.strip(),
            "format": "json",
            "limit": 1,
            "addressdetails": 0,
        }
        headers = {"User-Agent": _USER_AGENT}
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(_NOMINATIM_URL, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        if not data:
            logger.debug("Geocoding: kein Ergebnis für %r", place_name)
            return None

        first = data[0]
        result: dict[str, Any] = {
            "lat": float(first["lat"]),
            "lon": float(first["lon"]),
            "display_name": first.get("display_name", place_name),
        }

        with _cache_lock:
            _cache[key] = result
        logger.debug("Geocoding: %r → lat=%.4f, lon=%.4f", place_name, result["lat"], result["lon"])
        return result

    except Exception as exc:  # noqa: BLE001
        logger.warning("Geocoding fehlgeschlagen für %r: %s", place_name, exc)
        return None
