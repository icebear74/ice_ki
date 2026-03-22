"""
MySQL Connection Pool with automatic database + schema initialisation.

Startup sequence
----------------
1. Connect WITHOUT database parameter → create database if missing.
2. Connect WITH database parameter → create connection pool.
3. Run schema.sql if any of the expected tables are absent.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import mysql.connector
import mysql.connector.pooling

logger = logging.getLogger(__name__)

# Tables that schema.sql creates – used to decide whether to run it.
_EXPECTED_TABLES = {
    "users",
    "user_memory",
    "global_memory",
    "wiki_chunks",
    "knowledge_entries",
    "conversation_log",
}

_SCHEMA_FILE = Path(__file__).parent / "schema.sql"

_pool: mysql.connector.pooling.MySQLConnectionPool | None = None


def _get_mysql_cfg() -> dict:
    """Import config lazily so the module can be imported without config.py present."""
    try:
        import config  # noqa: PLC0415
        return dict(config.MYSQL)
    except ImportError:
        raise RuntimeError(
            "config.py not found. Copy config.py.example → config.py and fill in credentials."
        )


def _ensure_database(cfg: dict) -> None:
    """Create the database schema if it does not exist yet."""
    db_name = cfg["database"]
    init_cfg = {k: v for k, v in cfg.items() if k not in ("database", "pool_size")}
    try:
        conn = mysql.connector.connect(**init_cfg)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor.close()
        conn.close()
        logger.info("Database '%s' ready.", db_name)
    except mysql.connector.Error as exc:
        raise RuntimeError(
            f"Cannot connect to MySQL to create database '{db_name}'. "
            f"Check host/port/user/password in config.py and ensure the MySQL user "
            f"has CREATE privilege.\nMySQL error: {exc}"
        ) from exc


def _create_pool(cfg: dict) -> mysql.connector.pooling.MySQLConnectionPool:
    pool_cfg = {k: v for k, v in cfg.items() if k != "pool_size"}
    return mysql.connector.pooling.MySQLConnectionPool(
        pool_name="ice_brain",
        pool_size=cfg.get("pool_size", 5),
        **pool_cfg,
    )


def _tables_exist(pool: mysql.connector.pooling.MySQLConnectionPool, db_name: str) -> bool:
    conn = pool.get_connection()
    try:
        cursor = conn.cursor()
        # Use a parameterised query for db_name; table names come from the
        # hardcoded _EXPECTED_TABLES constant and are validated against an
        # allowlist before being included in the IN clause.
        allowed = {t for t in _EXPECTED_TABLES if t.replace("_", "").isalnum()}
        placeholders = ", ".join(["%s"] * len(allowed))
        cursor.execute(
            "SELECT TABLE_NAME FROM information_schema.TABLES "
            f"WHERE TABLE_SCHEMA = %s AND TABLE_NAME IN ({placeholders})",
            (db_name, *allowed),
        )
        found = {row[0] for row in cursor.fetchall()}
        cursor.close()
        return allowed.issubset(found)
    finally:
        conn.close()


def _run_schema(pool: mysql.connector.pooling.MySQLConnectionPool) -> None:
    sql = _SCHEMA_FILE.read_text(encoding="utf-8")
    # Split on semicolons to execute statement by statement.
    statements = [s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")]
    conn = pool.get_connection()
    try:
        cursor = conn.cursor()
        for stmt in statements:
            cursor.execute(stmt)
        conn.commit()
        cursor.close()
        logger.info("Schema applied from %s.", _SCHEMA_FILE)
    finally:
        conn.close()


def init_db() -> None:
    """Initialise the global connection pool and run schema if needed.

    Called once at server startup.
    """
    global _pool  # noqa: PLW0603
    cfg = _get_mysql_cfg()
    _ensure_database(cfg)
    _pool = _create_pool(cfg)
    if not _tables_exist(_pool, cfg["database"]):
        logger.info("Tables missing – running schema.sql …")
        _run_schema(_pool)
    else:
        logger.info("All tables present.")


@contextmanager
def get_connection() -> Generator[mysql.connector.MySQLConnection, None, None]:
    """Context manager that yields a pooled connection and returns it on exit."""
    if _pool is None:
        raise RuntimeError("DB pool not initialised. Call init_db() first.")
    conn = _pool.get_connection()
    try:
        yield conn
    finally:
        conn.close()
