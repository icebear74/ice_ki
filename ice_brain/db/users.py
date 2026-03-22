"""
Per-user settings and account management.

- User accounts are stored in the `users` table (user_id, username, password_hash, role).
- Per-user preferences (e.g. timezone) are stored in `user_memory`
  with category='timezone' and importance=1.0.
"""

from __future__ import annotations

import logging
import secrets
import string

import bcrypt

from db.connection import get_connection

logger = logging.getLogger(__name__)

_FALLBACK_TIMEZONE = "Europe/Berlin"
_TZ_CATEGORY = "timezone"
_TZ_IMPORTANCE = 1.0


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches *hashed*."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def _random_password(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return "".join(secrets.choice(alphabet) for _ in range(length))


# ---------------------------------------------------------------------------
# Account helpers
# ---------------------------------------------------------------------------

def create_user(username: str, password: str, role: str = "user") -> str:
    """Create a new user account.  Returns the new user_id.

    Raises an IntegrityError (mysql-connector) if the username already exists.
    """
    user_id = username.lower().replace(" ", "_")
    pw_hash = hash_password(password)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (user_id, username, password_hash, role) VALUES (%s, %s, %s, %s)",
            (user_id, username, pw_hash, role),
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
    If a new admin is created the generated password is printed once to the log.
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id FROM users WHERE role = 'admin' LIMIT 1")
            row = cursor.fetchone()
            cursor.close()
        if row:
            logger.info("Admin user already present: %r", row[0])
            return
        password = _random_password()
        create_user(admin_username, password, role="admin")
        logger.info("=" * 60)
        logger.info("ADMIN USER CREATED")
        logger.info("  Username : %s", admin_username)
        logger.info("  Password : %s", password)
        logger.info("  Bitte das Passwort sofort sichern und ändern!")
        logger.info("=" * 60)
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
