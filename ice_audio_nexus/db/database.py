"""
ice_audio_nexus – Datenbank-Verbindung und automatische Tabellenerstellung.

Beim ersten Start werden alle Tabellen automatisch angelegt (CREATE TABLE IF NOT EXISTS).
Zugangsdaten werden aus der .env-Datei gelesen.
"""

from __future__ import annotations

import logging
import os
import struct
from pathlib import Path
from typing import Optional

import mariadb
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# .env aus dem Projektverzeichnis laden
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


def get_connection() -> mariadb.Connection:
    """Erstellt eine neue MariaDB-Verbindung aus den .env-Konfigurationsdaten."""
    try:
        conn = mariadb.connect(
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            host=os.environ.get("DB_HOST", "localhost"),
            port=int(os.environ.get("DB_PORT", 3306)),
            database=os.environ["DB_NAME"],
            autocommit=False,
        )
        return conn
    except mariadb.Error as e:
        logger.error("Datenbankverbindung fehlgeschlagen: %s", e)
        raise


def _ddl_statements() -> list[str]:
    """Gibt alle CREATE TABLE IF NOT EXISTS Statements zurück."""
    return [
        # -----------------------------------------------------------------
        # voice_profiles – biometrischer Stimm-Fingerabdruck
        # VECTOR(512): Float32-Vektoren (~2KB) – Standard für PyAnnote
        # -----------------------------------------------------------------
        """
        CREATE TABLE IF NOT EXISTS voice_profiles (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            voice_vector    VECTOR(512) NOT NULL
                            COMMENT 'PyAnnote Float32-Embedding (512-dim)',
            sample_count    INT         NOT NULL DEFAULT 1
                            COMMENT 'Anzahl der gemittelten Samples',
            is_confirmed    BOOLEAN     NOT NULL DEFAULT FALSE
                            COMMENT 'Durch Nutzer bestaetigt?',
            created_at      TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        # -----------------------------------------------------------------
        # identities – Charakter in Serien-Kontext
        # Ein Synchronsprecher kann mehrere Identitäten haben.
        # -----------------------------------------------------------------
        """
        CREATE TABLE IF NOT EXISTS identities (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            voice_id        INT          NOT NULL
                            COMMENT 'Fremdschluessel auf voice_profiles',
            character_name  VARCHAR(255) NOT NULL
                            COMMENT 'z.B. Daryl Dixon',
            series_name     VARCHAR(255) NOT NULL
                            COMMENT 'z.B. The Walking Dead',
            sync_actor_name VARCHAR(255)
                            COMMENT 'Synchronsprecher (optional)',
            notes           TEXT         COMMENT 'Freitext-Notizen',
            created_at      TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (voice_id) REFERENCES voice_profiles(id)
                ON DELETE CASCADE,
            UNIQUE KEY uq_identity (character_name, series_name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        # -----------------------------------------------------------------
        # episode_segments – Timeline der Sprecher pro Episode
        # -----------------------------------------------------------------
        """
        CREATE TABLE IF NOT EXISTS episode_segments (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            series_name     VARCHAR(255) NOT NULL
                            COMMENT 'Serienname',
            episode_title   VARCHAR(255) NOT NULL
                            COMMENT 'Episodentitel oder Dateiname',
            video_path      VARCHAR(512)
                            COMMENT 'Relativer Pfad zur Quelldatei',
            start_ms        INT          NOT NULL
                            COMMENT 'Startzeit in Millisekunden',
            end_ms          INT          NOT NULL
                            COMMENT 'Endzeit in Millisekunden',
            raw_speaker_id  VARCHAR(64)  NOT NULL
                            COMMENT 'Temporaere Diarization-ID (z.B. SPEAKER_00)',
            identity_id     INT
                            COMMENT 'Zugeordnete Identitaet (NULL = unbekannt)',
            transcript      TEXT         COMMENT 'Whisper-Transkript des Segments',
            confidence      FLOAT        COMMENT 'Aehnlichkeits-Score (0.0-1.0)',
            is_confirmed    BOOLEAN      NOT NULL DEFAULT FALSE
                            COMMENT 'Durch Nutzer bestaetigt?',
            created_at      TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (identity_id) REFERENCES identities(id)
                ON DELETE SET NULL,
            INDEX idx_episode  (series_name, episode_title),
            INDEX idx_timeline (series_name, episode_title, start_ms),
            INDEX idx_speaker  (raw_speaker_id),
            INDEX idx_identity (identity_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
    ]


def init_db() -> None:
    """
    Stellt sicher, dass die Datenbank und alle Tabellen existieren.
    Wird beim ersten Start automatisch aufgerufen.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        for stmt in _ddl_statements():
            cur.execute(stmt.strip())
        conn.commit()
        logger.info("Datenbank initialisiert (alle Tabellen vorhanden).")
    except mariadb.Error as e:
        logger.error("Fehler bei der DB-Initialisierung: %s", e)
        conn.rollback()
        raise
    finally:
        conn.close()


# ------------------------------------------------------------------
# Hilfs-Funktionen für Vektor-Serialisierung
# ------------------------------------------------------------------

def vector_to_bytes(embedding: list[float]) -> bytes:
    """Konvertiert eine Liste von Float32-Werten in binäre Bytes für VECTOR(512)."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def bytes_to_vector(data: bytes) -> list[float]:
    """Konvertiert VECTOR(512)-Binärdaten zurück in eine Float32-Liste."""
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


# ------------------------------------------------------------------
# Sprecher-Profil-Funktionen
# ------------------------------------------------------------------

def upsert_voice_profile(
    conn: mariadb.Connection,
    embedding: list[float],
    sample_count: int = 1,
    is_confirmed: bool = False,
) -> int:
    """
    Legt ein neues voice_profile an und gibt die neue ID zurück.
    """
    cur = conn.cursor()
    vec_bytes = vector_to_bytes(embedding)
    cur.execute(
        """
        INSERT INTO voice_profiles (voice_vector, sample_count, is_confirmed)
        VALUES (VEC_FromText(%s), %s, %s)
        """,
        (_float_list_to_vec_text(embedding), sample_count, is_confirmed),
    )
    conn.commit()
    return cur.lastrowid


def update_master_vector(
    conn: mariadb.Connection,
    voice_id: int,
    new_embedding: list[float],
    sample_count: int,
) -> None:
    """Aktualisiert den Master-Vektor eines bestehenden voice_profiles."""
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE voice_profiles
        SET voice_vector = VEC_FromText(%s),
            sample_count = %s,
            is_confirmed = TRUE
        WHERE id = %s
        """,
        (_float_list_to_vec_text(new_embedding), sample_count, voice_id),
    )
    conn.commit()


def find_similar_voice(
    conn: mariadb.Connection,
    embedding: list[float],
    series_name: str,
    threshold: float = 0.85,
    limit: int = 1,
) -> list[dict]:
    """
    Sucht in der Datenbank nach dem ähnlichsten Stimm-Vektor
    innerhalb des angegebenen Serien-Kontexts.

    Gibt eine Liste von Treffern zurück mit:
      - identity_id, character_name, series_name, voice_id, distance
    """
    cur = conn.cursor()
    vec_text = _float_list_to_vec_text(embedding)
    cur.execute(
        """
        SELECT
            i.id            AS identity_id,
            i.character_name,
            i.series_name,
            i.voice_id,
            VEC_DISTANCE_EUCLIDEAN(vp.voice_vector, VEC_FromText(%s)) AS distance
        FROM identities i
        JOIN voice_profiles vp ON i.voice_id = vp.id
        WHERE i.series_name = %s
        ORDER BY distance ASC
        LIMIT %s
        """,
        (vec_text, series_name, limit),
    )
    rows = cur.fetchall()
    results = []
    for row in rows:
        dist = row[4] if row[4] is not None else 9999.0
        # Euclidean distance → cosine-like threshold: kleinerer Wert = ähnlicher
        if dist <= (1.0 - threshold):
            results.append(
                {
                    "identity_id": row[0],
                    "character_name": row[1],
                    "series_name": row[2],
                    "voice_id": row[3],
                    "distance": dist,
                    "confidence": max(0.0, 1.0 - dist),
                }
            )
    return results


# ------------------------------------------------------------------
# Segment-Funktionen
# ------------------------------------------------------------------

def insert_segment(
    conn: mariadb.Connection,
    series_name: str,
    episode_title: str,
    video_path: str,
    start_ms: int,
    end_ms: int,
    raw_speaker_id: str,
    identity_id: Optional[int] = None,
    transcript: Optional[str] = None,
    confidence: Optional[float] = None,
) -> int:
    """Fügt ein neues episode_segment ein und gibt die ID zurück."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO episode_segments
            (series_name, episode_title, video_path, start_ms, end_ms,
             raw_speaker_id, identity_id, transcript, confidence)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            series_name,
            episode_title,
            video_path,
            start_ms,
            end_ms,
            raw_speaker_id,
            identity_id,
            transcript,
            confidence,
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_segments_for_episode(
    conn: mariadb.Connection,
    series_name: str,
    episode_title: str,
) -> list[dict]:
    """Gibt alle Segmente einer Episode zurück, sortiert nach Startzeit."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            es.id, es.start_ms, es.end_ms, es.raw_speaker_id,
            es.identity_id, es.transcript, es.confidence, es.is_confirmed,
            i.character_name, i.series_name AS identity_series
        FROM episode_segments es
        LEFT JOIN identities i ON es.identity_id = i.id
        WHERE es.series_name = %s AND es.episode_title = %s
        ORDER BY es.start_ms ASC
        """,
        (series_name, episode_title),
    )
    cols = [
        "id", "start_ms", "end_ms", "raw_speaker_id",
        "identity_id", "transcript", "confidence", "is_confirmed",
        "character_name", "identity_series",
    ]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def assign_identity_to_speaker(
    conn: mariadb.Connection,
    series_name: str,
    episode_title: str,
    raw_speaker_id: str,
    identity_id: int,
    confirmed: bool = False,
) -> int:
    """
    Weist allen Segmenten mit raw_speaker_id in der Episode eine Identität zu.
    Gibt die Anzahl der aktualisierten Zeilen zurück.
    """
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE episode_segments
        SET identity_id = %s, is_confirmed = %s
        WHERE series_name = %s AND episode_title = %s AND raw_speaker_id = %s
        """,
        (identity_id, confirmed, series_name, episode_title, raw_speaker_id),
    )
    conn.commit()
    return cur.rowcount


def get_all_episodes(conn: mariadb.Connection) -> list[dict]:
    """Gibt alle bekannten Episoden (distinct) zurück."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT series_name, episode_title, video_path,
               MIN(created_at) AS scanned_at,
               COUNT(*) AS segment_count,
               SUM(CASE WHEN identity_id IS NOT NULL THEN 1 ELSE 0 END) AS assigned_count
        FROM episode_segments
        GROUP BY series_name, episode_title, video_path
        ORDER BY series_name, episode_title
        """
    )
    cols = ["series_name", "episode_title", "video_path", "scanned_at", "segment_count", "assigned_count"]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_all_identities(conn: mariadb.Connection) -> list[dict]:
    """Gibt alle bekannten Identitäten zurück."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT i.id, i.character_name, i.series_name, i.sync_actor_name,
               vp.sample_count, vp.is_confirmed
        FROM identities i
        JOIN voice_profiles vp ON i.voice_id = vp.id
        ORDER BY i.series_name, i.character_name
        """
    )
    cols = ["id", "character_name", "series_name", "sync_actor_name", "sample_count", "is_confirmed"]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ------------------------------------------------------------------
# Interne Hilfsfunktion
# ------------------------------------------------------------------

def _float_list_to_vec_text(embedding: list[float]) -> str:
    """Konvertiert eine Float-Liste in den VEC_FromText-kompatiblen String '[f1,f2,...]'."""
    return "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
