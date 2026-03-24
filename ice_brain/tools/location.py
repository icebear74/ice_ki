"""
Standort-Verwaltung – aktiven Standort des Benutzers aus user_memory lesen.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Regex to extract coordinates embedded in content text.
# Matches "📍 51.5672, 6.7331" or "📍51.5672,6.7331" etc.
_COORD_RE = re.compile(r"📍\s*(-?[\d.]+),\s*(-?[\d.]+)")


def get_active_location(user_id: str) -> dict[str, Any] | None:
    """Gibt den aktiven Standort des Benutzers zurück.

    Liest aus der user_memory-Tabelle (category='location') und priorisiert:
    1. Temporärer Standort (has expires_at, not expired) – z.B. Reise
    2. Permanenter Heimatstandort (expires_at IS NULL)

    Koordinaten werden per Regex aus dem Content-Text extrahiert (📍 lat, lon).
    Einträge ohne '📍'-Marker werden übersprungen (noch nicht geocoded).

    Rückgabe:
        {
            "content": str,
            "latitude": float,
            "longitude": float,
            "expires_at": datetime | None,
        }
        oder None wenn kein Standort gespeichert ist.
    """
    try:
        from db.connection import get_connection  # noqa: PLC0415

        with get_connection() as conn:
            cursor = conn.cursor()
            now_utc = datetime.now(tz=timezone.utc)

            # Lade Standort-Einträge die bereits mit 📍 geocoded wurden
            cursor.execute(
                """
                SELECT id, content, expires_at
                FROM user_memory
                WHERE user_id = %s
                  AND category = 'location'
                  AND content LIKE '%📍%'
                  AND (expires_at IS NULL OR expires_at > %s)
                ORDER BY
                    CASE WHEN expires_at IS NOT NULL THEN 0 ELSE 1 END ASC,
                    updated_at DESC
                LIMIT 10
                """,
                (user_id, now_utc),
            )
            rows = cursor.fetchall()
            cursor.close()

        if not rows:
            return None

        # Ersten gültigen Eintrag zurückgeben (temporär vor permanent)
        for row in rows:
            _row_id, content, expires_at = row

            # Koordinaten per Regex aus dem Content extrahieren
            coord_match = _COORD_RE.search(content or "")
            if not coord_match:
                continue

            try:
                lat = float(coord_match.group(1))
                lon = float(coord_match.group(2))
            except ValueError:
                continue

            # Abgelaufene temporäre Einträge überspringen
            if expires_at is not None:
                if isinstance(expires_at, datetime):
                    exp = expires_at
                    if exp.tzinfo is None:
                        exp = exp.replace(tzinfo=timezone.utc)
                    if exp <= now_utc:
                        continue
                else:
                    continue  # Ungültiger expires_at-Typ

            return {
                "content": content or "",
                "latitude": lat,
                "longitude": lon,
                "expires_at": expires_at,
            }

        return None

    except Exception as exc:  # noqa: BLE001
        logger.warning("get_active_location fehlgeschlagen für user %r: %s", user_id, exc)
        return None
