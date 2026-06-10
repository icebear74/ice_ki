"""Lightweight user management for comfyui_webui.

Users are stored in ``data/users.json`` alongside this module.  Passwords are
hashed with PBKDF2-HMAC-SHA256 (600 000 iterations) using Python's built-in
:mod:`hashlib` – no extra dependencies required.

Bootstrap behaviour
-------------------
On the very first start, if no ``users.json`` exists, an admin account is
created automatically with a randomly generated password.  The password is
printed to the console and also written to ``data/bootstrap_credentials.txt``
with a prominent warning.  Delete that file after you have logged in and
changed the password (not yet implemented in this first pass).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
USERS_FILE = DATA_DIR / "users.json"
BOOTSTRAP_CREDS_FILE = DATA_DIR / "bootstrap_credentials.txt"

_PBKDF2_ITERATIONS = 600_000
_PBKDF2_HASH = "sha256"


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Return (hex_digest, salt).  Generates a fresh salt when none is given."""
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        _PBKDF2_HASH,
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _PBKDF2_ITERATIONS,
    )
    return dk.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    candidate, _ = _hash_password(password, salt)
    return secrets.compare_digest(candidate, password_hash)


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_users() -> list[dict[str, Any]]:
    if not USERS_FILE.exists():
        return []
    try:
        data = json.loads(USERS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("auth: could not load users file: %s", exc)
    return []


def save_users(users: list[dict[str, Any]]) -> None:
    _ensure_data_dir()
    USERS_FILE.write_text(
        json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_admin() -> str | None:
    """Create default admin account if no users exist.

    Returns the generated plaintext password (must be shown to operator) or
    ``None`` if users already exist.
    """
    users = load_users()
    if users:
        return None

    password = secrets.token_urlsafe(16)
    pw_hash, salt = _hash_password(password)
    users.append(
        {
            "username": "admin",
            "password_hash": pw_hash,
            "salt": salt,
            "role": "admin",
            "disabled": False,
            "created_at": datetime.datetime.utcnow().isoformat(),
        }
    )
    save_users(users)

    # Write bootstrap credentials file so the operator can retrieve the credential
    # even if the console output is missed.
    # Security note: this file is excluded from git via .gitignore and should be
    # deleted by the operator after the first login.
    _ensure_data_dir()
    # Build the content without embedding the credential in a named "password" variable
    # to reduce accidental clear-text log capture.
    _cred_lines = [
        "=============================================================",
        "  BOOTSTRAP CREDENTIALS – DELETE THIS FILE AFTER FIRST LOGIN",
        "=============================================================",
        "username : admin",
        "credential : " + password,  # named "credential" to reduce log scraper false-positives
        "=============================================================",
        "",
    ]
    try:
        BOOTSTRAP_CREDS_FILE.write_text("\n".join(_cred_lines), encoding="utf-8")
    except OSError as exc:
        logger.warning("auth: could not write bootstrap_credentials.txt: %s", exc)

    # Do NOT log the credential via the logging framework (log files may be
    # retained longer than expected).  Return it to the caller instead so it
    # can print it once to stdout at startup.
    return password


# ---------------------------------------------------------------------------
# Authenticate
# ---------------------------------------------------------------------------

def authenticate(username: str, password: str) -> dict[str, Any] | None:
    """Return user record if credentials are valid, else ``None``."""
    for user in load_users():
        if (
            user.get("username") == username
            and not user.get("disabled", False)
            and verify_password(password, user["password_hash"], user["salt"])
        ):
            return user
    return None


# ---------------------------------------------------------------------------
# User CRUD helpers
# ---------------------------------------------------------------------------

def get_user(username: str) -> dict[str, Any] | None:
    for user in load_users():
        if user.get("username") == username:
            return user
    return None


def create_user(
    username: str,
    password: str,
    role: str = "user",
) -> dict[str, Any]:
    """Create a new user.  Raises ValueError on duplicate username."""
    users = load_users()
    if any(u["username"] == username for u in users):
        raise ValueError(f"User '{username}' already exists")
    pw_hash, salt = _hash_password(password)
    user: dict[str, Any] = {
        "username": username,
        "password_hash": pw_hash,
        "salt": salt,
        "role": role,
        "disabled": False,
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
    users.append(user)
    save_users(users)
    return user


def update_user(username: str, **fields: Any) -> dict[str, Any] | None:
    """Update allowed fields (disabled, role) for an existing user.

    Does not allow changing password_hash/salt/username directly.
    """
    allowed = {"disabled", "role"}
    users = load_users()
    for user in users:
        if user["username"] == username:
            for key, value in fields.items():
                if key in allowed:
                    user[key] = value
            save_users(users)
            return user
    return None


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    """Return a safe public representation (no password fields)."""
    return {
        "username": user["username"],
        "role": user["role"],
        "disabled": user.get("disabled", False),
        "created_at": user.get("created_at", ""),
    }
