"""
Standort-Verwaltung – aktiven Standort des Benutzers aus user_memory lesen.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def get_active_location(user_id: str) -> dict[str, Any] | None:
    """Gibt den aktiven Standort des Benutzers zurück.

    Liest aus der user_memory-Tabelle (category='location') und priorisiert:
    1. Temporärer Standort (has expires_at, not expired) – z.B. Reise
    2. Permanenter Heimatstandort (expires_at IS NULL)

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

            # Alle Standort-Einträge mit gültigen Koordinaten laden
            cursor.execute(
                """
                SELECT content, latitude, longitude, expires_at
                FROM user_memory
                WHERE user_id = %s
                  AND category = 'location'
                  AND latitude IS NOT NULL
                  AND longitude IS NOT NULL
                ORDER BY
                    CASE WHEN expires_at IS NOT NULL AND expires_at > %s THEN 0 ELSE 1 END ASC,
                    created_at DESC
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
            content, lat, lon, expires_at = row
            if lat is None or lon is None:
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
                "latitude": float(lat),
                "longitude": float(lon),
                "expires_at": expires_at,
            }

        return None

    except Exception as exc:  # noqa: BLE001
        logger.warning("get_active_location fehlgeschlagen für user %r: %s", user_id, exc)
        return None
