"""
ice_audio_nexus – database.py
Auto-creates all required tables in MariaDB 11.7 on first run.

Schema (Multi-Vector Identity):
  actors           – one row per real-world voice / actor (e.g. 'Patrick Stewart')
  identities       – one row per character / persona (e.g. 'Jean-Luc Picard'),
                     linked to an actor via actor_id + context_filter
  voice_samples    – n rows per identity; each holds a VECTOR(512) embedding
                     plus metadata (context, confirmed flag, timestamps)
  episode_segments – timeline of detected speaker segments per episode
"""

import os
import struct
import logging

import mariadb
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = [
    # 0. Actors – the real-world voice / person behind the characters
    """
    CREATE TABLE IF NOT EXISTS actors (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        name       VARCHAR(255) NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                                      ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_actor_name (name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    # 1. Identity anchor – the person / character
    """
    CREATE TABLE IF NOT EXISTS identities (
        id             INT AUTO_INCREMENT PRIMARY KEY,
        name           VARCHAR(255) NOT NULL,
        description    TEXT,
        actor_id       INT DEFAULT NULL,
        context_filter VARCHAR(255) DEFAULT NULL
                       COMMENT 'SQL LIKE pattern for context matching, e.g. Star Trek%',
        created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                                          ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_identity_name (name),
        FOREIGN KEY (actor_id) REFERENCES actors(id) ON DELETE SET NULL,
        INDEX idx_identity_actor (actor_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    # 2. Voice samples – many per identity, each with its own 512-dim vector
    """
    CREATE TABLE IF NOT EXISTS voice_samples (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        identity_id INT NOT NULL,
        embedding   VECTOR(512) NOT NULL,
        context     VARCHAR(255) DEFAULT NULL COMMENT 'e.g. TNG Season 1, Picard S3E02',
        is_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
        created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                                       ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (identity_id) REFERENCES identities(id) ON DELETE CASCADE,
        INDEX idx_vs_identity (identity_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    # 3. Episode segments – speaker timeline with link to matched identity
    """
    CREATE TABLE IF NOT EXISTS episode_segments (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        series_name     VARCHAR(255),
        episode_title   VARCHAR(255),
        video_path      TEXT,
        start_ms        INT NOT NULL,
        end_ms          INT NOT NULL,
        speaker_label   VARCHAR(100) COMMENT 'Temporary diarization label (SPEAKER_01)',
        embedding       VECTOR(512) NULL COMMENT 'Raw speaker embedding from diarization',
        identity_id     INT DEFAULT NULL,
        matched_sample_id INT DEFAULT NULL COMMENT 'Which voice_sample triggered the match',
        match_distance  FLOAT DEFAULT NULL COMMENT 'Cosine distance of the winning match',
        transcript      TEXT,
        confidence      FLOAT DEFAULT NULL,
        is_suggestion   BOOLEAN NOT NULL DEFAULT FALSE
                        COMMENT 'True = proposed match (slightly high distance), needs user confirmation',
        created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                                           ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (identity_id)      REFERENCES identities(id)     ON DELETE SET NULL,
        FOREIGN KEY (matched_sample_id) REFERENCES voice_samples(id)  ON DELETE SET NULL,
        INDEX idx_seg_episode (series_name, episode_title),
        INDEX idx_seg_identity (identity_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _get_conn_params() -> dict:
    return {
        "host":     os.getenv("DB_HOST", "localhost"),
        "port":     int(os.getenv("DB_PORT", "3306")),
        "user":     os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME", "ice_nexus_db"),
    }


def get_connection() -> mariadb.Connection:
    params = _get_conn_params()
    return mariadb.connect(**params)


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

def ensure_schema() -> None:
    """Create all tables if they do not exist yet. Called on application start."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        for ddl in _DDL:
            cur.execute(ddl)
        # Migrate pre-existing episode_segments tables that were created before
        # the embedding column was added.
        cur.execute(
            """
            ALTER TABLE episode_segments
            ADD COLUMN IF NOT EXISTS embedding VECTOR(512) NULL
                COMMENT 'Raw speaker embedding from diarization'
            """
        )
        conn.commit()
        logger.info("Database schema verified / created.")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Vector serialisation helpers (MariaDB VECTOR uses raw IEEE 754 float32)
# ---------------------------------------------------------------------------

def vector_to_bytes(embedding: list[float]) -> bytes:
    """Encode a Python list of floats → raw bytes for VECTOR(512) column."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def bytes_to_vector(raw: bytes) -> list[float]:
    """Decode raw bytes from VECTOR(512) column → Python list of floats."""
    n = len(raw) // 4
    return list(struct.unpack(f"{n}f", raw))


# ---------------------------------------------------------------------------
# Identity CRUD
# ---------------------------------------------------------------------------

def list_identities(conn: mariadb.Connection) -> list[dict]:
    cur = conn.cursor()
    cur.execute("""
        SELECT i.id, i.name, i.description,
               COUNT(vs.id) AS sample_count
        FROM identities i
        LEFT JOIN voice_samples vs ON vs.identity_id = i.id
        GROUP BY i.id
        ORDER BY i.name
    """)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_identity(conn: mariadb.Connection, identity_id: int) -> dict | None:
    cur = conn.cursor()
    cur.execute("SELECT id, name, description FROM identities WHERE id = ?", (identity_id,))
    row = cur.fetchone()
    if row is None:
        return None
    return {"id": row[0], "name": row[1], "description": row[2]}


def create_identity(conn: mariadb.Connection, name: str, description: str = "") -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO identities (name, description) VALUES (?, ?)",
        (name, description),
    )
    conn.commit()
    return cur.lastrowid


def update_identity(conn: mariadb.Connection, identity_id: int, name: str, description: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "UPDATE identities SET name = ?, description = ? WHERE id = ?",
        (name, description, identity_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Actor CRUD
# ---------------------------------------------------------------------------

def list_actors(conn: mariadb.Connection) -> list[dict]:
    cur = conn.cursor()
    cur.execute("""
        SELECT a.id, a.name,
               COUNT(i.id) AS identity_count,
               a.created_at, a.updated_at
        FROM actors a
        LEFT JOIN identities i ON i.actor_id = a.id
        GROUP BY a.id
        ORDER BY a.name
    """)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_actor(conn: mariadb.Connection, actor_id: int) -> dict | None:
    cur = conn.cursor()
    cur.execute("SELECT id, name, created_at, updated_at FROM actors WHERE id = ?", (actor_id,))
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def create_actor(conn: mariadb.Connection, name: str) -> int:
    cur = conn.cursor()
    cur.execute("INSERT INTO actors (name) VALUES (?)", (name,))
    conn.commit()
    return cur.lastrowid


def update_actor(conn: mariadb.Connection, actor_id: int, name: str) -> None:
    cur = conn.cursor()
    cur.execute("UPDATE actors SET name = ? WHERE id = ?", (name, actor_id))
    conn.commit()


# ---------------------------------------------------------------------------
# Voice sample CRUD
# ---------------------------------------------------------------------------

def add_voice_sample(
    conn: mariadb.Connection,
    identity_id: int,
    embedding: list[float],
    context: str = "",
    is_confirmed: bool = False,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO voice_samples (identity_id, embedding, context, is_confirmed)
           VALUES (?, ?, ?, ?)""",
        (identity_id, vector_to_bytes(embedding), context or None, is_confirmed),
    )
    conn.commit()
    return cur.lastrowid


def list_voice_samples(conn: mariadb.Connection, identity_id: int) -> list[dict]:
    cur = conn.cursor()
    cur.execute(
        """SELECT id, identity_id, embedding, context, is_confirmed, created_at
           FROM voice_samples WHERE identity_id = ? ORDER BY created_at""",
        (identity_id,),
    )
    results = []
    for row in cur.fetchall():
        results.append({
            "id":           row[0],
            "identity_id":  row[1],
            "embedding":    bytes_to_vector(row[2]),
            "context":      row[3],
            "is_confirmed": bool(row[4]),
            "created_at":   str(row[5]),
        })
    return results


def confirm_voice_sample(conn: mariadb.Connection, sample_id: int) -> None:
    cur = conn.cursor()
    cur.execute("UPDATE voice_samples SET is_confirmed = TRUE WHERE id = ?", (sample_id,))
    conn.commit()


def delete_voice_sample(conn: mariadb.Connection, sample_id: int) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM voice_samples WHERE id = ?", (sample_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Nearest-neighbour vector search
# ---------------------------------------------------------------------------

def find_nearest_identity(
    conn: mariadb.Connection,
    embedding: list[float],
    match_threshold: float = 0.25,
    suggest_threshold: float = 0.45,
) -> dict:
    """
    Search all voice_samples using VEC_DISTANCE_COSINE (MariaDB 11.7+) and
    return the closest match.

    Returns a dict with keys:
      status        – 'matched' | 'suggest' | 'unknown'
      identity_id   – int or None
      identity_name – str or None
      sample_id     – int or None   (which sample triggered the match)
      sample_context– str or None
      distance      – float or None
    """
    vec_bytes = vector_to_bytes(embedding)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT vs.id,
               vs.identity_id,
               i.name,
               vs.context,
               VEC_DISTANCE_COSINE(vs.embedding, ?) AS dist
        FROM voice_samples vs
        JOIN identities i ON i.id = vs.identity_id
        ORDER BY dist ASC
        LIMIT 1
        """,
        (vec_bytes,),
    )
    row = cur.fetchone()
    if row is None:
        return {"status": "unknown", "identity_id": None, "identity_name": None,
                "sample_id": None, "sample_context": None, "distance": None}

    sample_id, identity_id, identity_name, sample_context, distance = row
    if distance <= match_threshold:
        return {
            "status": "matched",
            "identity_id": identity_id,
            "identity_name": identity_name,
            "sample_id": sample_id,
            "sample_context": sample_context,
            "distance": float(distance),
        }
    if distance <= suggest_threshold:
        return {
            "status": "suggest",
            "identity_id": identity_id,
            "identity_name": identity_name,
            "sample_id": sample_id,
            "sample_context": sample_context,
            "distance": float(distance),
        }
    return {"status": "unknown", "identity_id": None, "identity_name": None,
            "sample_id": None, "sample_context": None, "distance": float(distance)}


# ---------------------------------------------------------------------------
# Episode segment helpers
# ---------------------------------------------------------------------------

def upsert_segment(conn: mariadb.Connection, **kwargs) -> int:
    """Insert a new episode segment row.  kwargs map directly to column names."""
    # Explicit allowlist of valid column names – guards against SQL injection
    # if the caller ever passes unexpected keys.
    _ALLOWED_COLS = (
        "series_name", "episode_title", "video_path", "start_ms", "end_ms",
        "speaker_label", "embedding", "identity_id", "matched_sample_id",
        "match_distance", "transcript", "confidence", "is_suggestion",
    )
    data = {k: v for k, v in kwargs.items() if k in _ALLOWED_COLS}
    # Build column list from the verified allowlist (not from caller input)
    cols         = ", ".join(data.keys())
    placeholders = ", ".join("?" for _ in data)
    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO episode_segments ({cols}) VALUES ({placeholders})",  # noqa: S608
        list(data.values()),
    )
    conn.commit()
    return cur.lastrowid


def update_segment_identity(
    conn: mariadb.Connection,
    segment_id: int,
    identity_id: int,
    matched_sample_id: int | None = None,
    match_distance: float | None = None,
    is_suggestion: bool = False,
) -> None:
    cur = conn.cursor()
    cur.execute(
        """UPDATE episode_segments
           SET identity_id = ?, matched_sample_id = ?,
               match_distance = ?, is_suggestion = ?
           WHERE id = ?""",
        (identity_id, matched_sample_id, match_distance, is_suggestion, segment_id),
    )
    conn.commit()


def get_segment_embedding(
    conn: mariadb.Connection,
    segment_id: int,
) -> list[float] | None:
    """Return the stored speaker embedding for a segment, or None if absent."""
    cur = conn.cursor()
    cur.execute(
        "SELECT embedding FROM episode_segments WHERE id = ?",
        (segment_id,),
    )
    row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    return bytes_to_vector(row[0])


def list_processed_episodes(conn: mariadb.Connection) -> list[dict]:
    """
    Return a grouped list of all episodes that have been processed by the scanner.
    Each entry contains series_name, episode_title, the stored video_path and the
    total segment count – enough for the Web UI library view.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT
            series_name,
            episode_title,
            MIN(video_path)  AS video_path,
            COUNT(*)         AS segment_count
        FROM episode_segments
        WHERE series_name IS NOT NULL
          AND episode_title IS NOT NULL
        GROUP BY series_name, episode_title
        ORDER BY series_name, episode_title
    """)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]



def get_episode_segments(
    conn: mariadb.Connection,
    series_name: str,
    episode_title: str,
) -> list[dict]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            es.id, es.start_ms, es.end_ms, es.speaker_label,
            es.identity_id, i.name AS identity_name,
            es.matched_sample_id, vs.context AS matched_sample_context,
            es.match_distance, es.transcript, es.confidence, es.is_suggestion
        FROM episode_segments es
        LEFT JOIN identities i    ON i.id  = es.identity_id
        LEFT JOIN voice_samples vs ON vs.id = es.matched_sample_id
        WHERE es.series_name = ? AND es.episode_title = ?
        ORDER BY es.start_ms
        """,
        (series_name, episode_title),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]
