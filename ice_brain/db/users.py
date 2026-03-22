"""
Per-user settings stored in the `users` table.

Currently manages:
  - timezone  (IANA name, e.g. "Europe/Berlin")
"""

from __future__ import annotations

import logging

from db.connection import get_connection

logger = logging.getLogger(__name__)

_FALLBACK_TIMEZONE = "Europe/Berlin"


def get_user_timezone(user_id: str) -> str:
    """Return the stored timezone for *user_id*, or the fallback if none is set."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT timezone FROM users WHERE user_id = %s",
                (user_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            return row[0] if row else _FALLBACK_TIMEZONE
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read timezone for user %r: %s", user_id, exc)
        return _FALLBACK_TIMEZONE


def upsert_user_timezone(user_id: str, timezone: str) -> None:
    """Persist *timezone* for *user_id* (insert or update)."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO users (user_id, timezone)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE timezone = VALUES(timezone)
                """,
                (user_id, timezone),
            )
            conn.commit()
            cursor.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save timezone for user %r: %s", user_id, exc)
