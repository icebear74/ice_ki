"""
Per-user settings and account management.

- User accounts are stored in the `users` table (user_id, username, role).
- Per-user preferences (e.g. timezone) are stored in `user_memory`
  with category='timezone' and importance=1.0.
"""

from __future__ import annotations

import logging
import uuid

from db.connection import get_connection

logger = logging.getLogger(__name__)

_FALLBACK_TIMEZONE = "Europe/Berlin"
_TZ_CATEGORY = "timezone"
_TZ_IMPORTANCE = 1.0


# ---------------------------------------------------------------------------
# Account helpers
# ---------------------------------------------------------------------------

def create_user(username: str, role: str = "user") -> str:
    """Create a new user account.  Returns the new user_id.

    Raises ValueError if the username already exists.
    """
    user_id = username.lower().replace(" ", "_")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (user_id, username, role) VALUES (%s, %s, %s)",
            (user_id, username, role),
        )
        conn.commit()
        cursor.close()
    return user_id


def user_exists(user_id: str) -> bool:
    """Return True if *user_id* has a row in the users table."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM users WHERE user_id = %s", (user_id,))
            found = cursor.fetchone() is not None
            cursor.close()
            return found
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not check user existence for %r: %s", user_id, exc)
        return False


def ensure_admin_user(admin_username: str = "admin") -> None:
    """Create the admin user if no admin account exists yet.

    Called once at server startup.  Idempotent – safe to call on every restart.
    """
    try:
        admin_id = admin_username.lower().replace(" ", "_")
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id FROM users WHERE role = 'admin' LIMIT 1")
            row = cursor.fetchone()
            cursor.close()
        if row:
            logger.info("Admin user already present: %r", row[0])
            return
        create_user(admin_username, role="admin")
        logger.info("Admin user created: %r (user_id=%r)", admin_username, admin_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not ensure admin user: %s", exc)


# ---------------------------------------------------------------------------
# Timezone helpers  (stored in user_memory, category='timezone')
# ---------------------------------------------------------------------------

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
