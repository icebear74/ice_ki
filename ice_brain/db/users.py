"""
Per-user settings and account management.

- User accounts are stored in the `users` table
  (user_id, username, password_hash, role).
  password_hash = NULL means the account was just created and the user
  must set a password on first login.
- Per-user preferences (e.g. timezone) are stored in `user_memory`
  with category='timezone' and importance=1.0.
"""

from __future__ import annotations

import logging

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


# ---------------------------------------------------------------------------
# Account helpers
# ---------------------------------------------------------------------------

def create_user(username: str, role: str = "user") -> str:
    """Create a new user account with no password set (first-login state).

    Returns the new user_id.
    Raises an IntegrityError (mysql-connector) if the username already exists.
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


def is_first_login(user_id: str) -> bool:
    """Return True when the user has not yet set a password (password_hash IS NULL)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT password_hash FROM users WHERE user_id = %s", (user_id,)
        )
        row = cursor.fetchone()
        cursor.close()
    return row is not None and row[0] is None


def set_password(user_id: str, plain: str) -> None:
    """Hash *plain* and store it for *user_id*."""
    pw_hash = hash_password(plain)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password_hash = %s WHERE user_id = %s",
            (pw_hash, user_id),
        )
        conn.commit()
        cursor.close()


def authenticate(username: str, password: str) -> "tuple[str, bool] | None":
    """Check credentials.  Returns (user_id, first_login) or None if invalid.

    - Returns (user_id, True)  when password_hash IS NULL (first login – set password).
    - Returns (user_id, False) when password matches stored hash.
    - Returns None when the username doesn't exist or the password is wrong.
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_id, password_hash FROM users WHERE username = %s",
                (username,),
            )
            row = cursor.fetchone()
            cursor.close()
        if row is None:
            return None
        user_id, pw_hash = row
        if pw_hash is None:
            # First login: no password set yet – allow through so UI can set one.
            return (user_id, True)
        if verify_password(password, pw_hash):
            return (user_id, False)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Authentication error for %r: %s", username, exc)
        return None


def ensure_admin_user(admin_username: str = "admin") -> None:
    """Create the admin user (without a password) if no admin account exists yet.

    Called once at server startup.  Idempotent – safe to call on every restart.
    The admin must set their password on first login.
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
        create_user(admin_username, role="admin")
        logger.info(
            "Admin user %r created.  Passwort wird beim ersten Login gesetzt.",
            admin_username,
        )
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
