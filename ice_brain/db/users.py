"""
Per-user settings stored as rows in the `user_memory` table.

Timezone is stored with category='timezone' and importance=1.0 so it is
treated as a high-priority permanent preference.
"""

from __future__ import annotations

import logging

from db.connection import get_connection

logger = logging.getLogger(__name__)

_FALLBACK_TIMEZONE = "Europe/Berlin"
_TZ_CATEGORY = "timezone"
_TZ_IMPORTANCE = 1.0


def get_user_timezone(user_id: str) -> str:
    """Return the stored timezone for *user_id*, or the fallback if none is set."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content FROM user_memory "
                "WHERE user_id = %s AND category = %s "
                "ORDER BY updated_at DESC LIMIT 1",
                (user_id, _TZ_CATEGORY),
            )
            row = cursor.fetchone()
            cursor.close()
            return row[0] if row else _FALLBACK_TIMEZONE
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read timezone for user %r: %s", user_id, exc)
        return _FALLBACK_TIMEZONE


def upsert_user_timezone(user_id: str, timezone: str) -> None:
    """Persist *timezone* for *user_id* (insert or update in user_memory)."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE user_memory SET content = %s, updated_at = NOW() "
                "WHERE user_id = %s AND category = %s",
                (timezone, user_id, _TZ_CATEGORY),
            )
            if cursor.rowcount == 0:
                cursor.execute(
                    "INSERT INTO user_memory (user_id, category, content, importance) "
                    "VALUES (%s, %s, %s, %s)",
                    (user_id, _TZ_CATEGORY, timezone, _TZ_IMPORTANCE),
                )
            conn.commit()
            cursor.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save timezone for user %r: %s", user_id, exc)
