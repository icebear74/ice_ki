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


def _missing_tables(pool: mysql.connector.pooling.MySQLConnectionPool, db_name: str) -> set[str]:
    """Return the subset of _EXPECTED_TABLES that are not yet present in the DB."""
    conn = pool.get_connection()
    try:
        cursor = conn.cursor()
        allowed = {t for t in _EXPECTED_TABLES if t.replace("_", "").isalnum()}
        placeholders = ", ".join(["%s"] * len(allowed))
        cursor.execute(
            "SELECT TABLE_NAME FROM information_schema.TABLES "
            f"WHERE TABLE_SCHEMA = %s AND TABLE_NAME IN ({placeholders})",
            (db_name, *allowed),
        )
        found = {row[0] for row in cursor.fetchall()}
        cursor.close()
        return allowed - found
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
            try:
                cursor.execute(stmt)
            except mysql.connector.Error as exc:
                logger.warning("Schema statement skipped (%s): %.120s …", exc.errno, stmt[:120])
        conn.commit()
        cursor.close()
        logger.info("Schema applied from %s.", _SCHEMA_FILE)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# MySQL 8.4 VECTOR column + index migrations
# ---------------------------------------------------------------------------

_VECTOR_COLUMNS: list[tuple[str, str]] = [
    ("wiki_chunks",       "embedding"),
    ("knowledge_entries", "embedding"),
]

_VECTOR_INDEXES: list[tuple[str, str, str]] = [
    # (index_name, table_name, column_name)
    ("idx_wiki_embedding",       "wiki_chunks",       "embedding"),
    ("idx_knowledge_embedding",  "knowledge_entries", "embedding"),
]


def _column_exists(cursor: mysql.connector.cursor.MySQLCursor, db_name: str, table: str, column: str) -> bool:
    cursor.execute(
        "SELECT 1 FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s",
        (db_name, table, column),
    )
    return cursor.fetchone() is not None


def _index_exists(cursor: mysql.connector.cursor.MySQLCursor, db_name: str, table: str, index: str) -> bool:
    cursor.execute(
        "SELECT 1 FROM information_schema.STATISTICS "
        "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND INDEX_NAME = %s",
        (db_name, table, index),
    )
    return cursor.fetchone() is not None


def _ensure_vector_columns(pool: mysql.connector.pooling.MySQLConnectionPool, db_name: str) -> None:
    """ADD COLUMN embedding VECTOR(768) to tables that are missing it (existing installs)."""
    conn = pool.get_connection()
    try:
        cursor = conn.cursor()
        for table, column in _VECTOR_COLUMNS:
            if not _column_exists(cursor, db_name, table, column):
                try:
                    cursor.execute(
                        f"ALTER TABLE `{table}` ADD COLUMN `{column}` VECTOR(768) NULL "
                        f"COMMENT 'Text-Embedding 768-dim; NULL bis verarbeitet'"
                    )
                    logger.info("VECTOR column %s.%s added.", table, column)
                except mysql.connector.Error as exc:
                    logger.warning("Could not add VECTOR column %s.%s: %s", table, column, exc)
        cursor.close()
    finally:
        conn.close()


def _ensure_vector_indexes(pool: mysql.connector.pooling.MySQLConnectionPool, db_name: str) -> None:
    """CREATE VECTOR INDEX (HNSW) where missing.  Requires MySQL 8.4+.

    Each index creation is attempted independently so a single failure does not
    block the others.  Indexes on columns with only NULL values will fail –
    they will be retried on the next startup once embeddings are populated.
    """
    conn = pool.get_connection()
    try:
        cursor = conn.cursor()
        for idx_name, table, column in _VECTOR_INDEXES:
            if _index_exists(cursor, db_name, table, idx_name):
                continue
            # Only attempt if the column has at least one non-NULL value.
            cursor.execute(
                f"SELECT 1 FROM `{table}` WHERE `{column}` IS NOT NULL LIMIT 1"
            )
            if cursor.fetchone() is None:
                logger.debug("Skipping VECTOR INDEX %s – no embeddings yet.", idx_name)
                continue
            try:
                cursor.execute(
                    f"CREATE VECTOR INDEX `{idx_name}` ON `{table}`(`{column}`) USING HNSW"
                )
                logger.info("VECTOR INDEX %s created on %s.%s.", idx_name, table, column)
            except mysql.connector.Error as exc:
                logger.warning(
                    "Could not create VECTOR INDEX %s (MySQL 8.4+ required): %s", idx_name, exc
                )
        cursor.close()
    finally:
        conn.close()


def init_db() -> None:
    """Initialise the global connection pool and run schema if needed.

    Called once at server startup.  Each missing table is created individually
    so that newly added tables are picked up on existing databases without
    requiring a full schema reset.

    After schema init:
    - _ensure_vector_columns: adds VECTOR columns to existing tables (migration).
    - _ensure_vector_indexes: creates HNSW indexes once embeddings are present.
    """
    global _pool  # noqa: PLW0603
    cfg = _get_mysql_cfg()
    _ensure_database(cfg)
    _pool = _create_pool(cfg)
    missing = _missing_tables(_pool, cfg["database"])
    if missing:
        logger.info("Missing tables %s – running schema.sql …", missing)
        _run_schema(_pool)
    else:
        logger.info("All tables present.")
    # Always run migrations (idempotent) – adds VECTOR columns to existing tables.
    _ensure_vector_columns(_pool, cfg["database"])
    # Create HNSW indexes where embeddings are already populated.
    _ensure_vector_indexes(_pool, cfg["database"])


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
