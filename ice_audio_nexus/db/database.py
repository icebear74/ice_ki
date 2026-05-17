"""
ice_audio_nexus – visual-first Step-1 persistence layer.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any

import mariadb
from dotenv import load_dotenv

load_dotenv()

SEED_WORKFLOW_STAGES = {"seed_discovery", "review", "finished", "expansion"}
SEED_REVIEW_STATES = {"pending", "confirmed", "needs_split", "ignored", "irrelevant"}
SEED_EXPANSION_STATES = {"blocked", "ready", "running", "done"}
_UNSET = object()
logger = logging.getLogger(__name__)


DDL_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS actors (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        name        VARCHAR(255) NOT NULL,
        description TEXT NULL,
        created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_actor_name (name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS voice_actors (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        name        VARCHAR(255) NOT NULL,
        notes       TEXT NULL,
        created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_voice_actor_name (name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS productions (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        title           VARCHAR(255) NOT NULL,
        production_type ENUM('series','movie','other') NOT NULL DEFAULT 'series',
        season_label    VARCHAR(64) NULL,
        metadata_json   LONGTEXT NULL,
        created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_production_title (title)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS roles (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        name        VARCHAR(255) NOT NULL,
        description TEXT NULL,
        created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_role_name (name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS actor_roles (
        id            INT AUTO_INCREMENT PRIMARY KEY,
        actor_id      INT NOT NULL,
        production_id INT NULL,
        role_id       INT NOT NULL,
        notes         TEXT NULL,
        created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (actor_id) REFERENCES actors(id) ON DELETE CASCADE,
        FOREIGN KEY (production_id) REFERENCES productions(id) ON DELETE CASCADE,
        FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE,
        UNIQUE KEY uq_actor_role (actor_id, production_id, role_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS videos (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        production_id   INT NULL,
        title           VARCHAR(255) NOT NULL,
        episode_code    VARCHAR(64) NULL,
        video_path      TEXT NOT NULL,
        duration_ms     INT NULL,
        scan_status     ENUM('pending','scanning','completed','failed') NOT NULL DEFAULT 'pending',
        last_scanned_at TIMESTAMP NULL,
        metadata_json   LONGTEXT NULL,
        created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (production_id) REFERENCES productions(id) ON DELETE SET NULL,
        UNIQUE KEY uq_video_path (video_path(512)),
        INDEX idx_video_prod (production_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS face_tracks (
        id                   INT AUTO_INCREMENT PRIMARY KEY,
        video_id             INT NOT NULL,
        start_ms             INT NOT NULL,
        end_ms               INT NOT NULL,
        frame_count          INT NOT NULL,
        mean_face_area       FLOAT NULL,
        mean_sharpness       FLOAT NULL,
        mean_confidence      FLOAT NULL,
        stability_score      FLOAT NULL,
        quality_score        FLOAT NULL,
        relevance_score      FLOAT NULL,
        is_clear             BOOLEAN NOT NULL DEFAULT FALSE,
        status               ENUM('candidate','assigned','ignored','unknown','background') NOT NULL DEFAULT 'candidate',
        assigned_actor_id    INT NULL,
        assigned_role_id     INT NULL,
        assignment_source    ENUM('manual','rematch','system') NULL,
        match_actor_id       INT NULL,
        match_score          FLOAT NULL,
        representative_image_path TEXT NULL,
        embedding_json       LONGTEXT NULL,
        metadata_json        LONGTEXT NULL,
        created_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
        FOREIGN KEY (assigned_actor_id) REFERENCES actors(id) ON DELETE SET NULL,
        FOREIGN KEY (assigned_role_id) REFERENCES roles(id) ON DELETE SET NULL,
        FOREIGN KEY (match_actor_id) REFERENCES actors(id) ON DELETE SET NULL,
        INDEX idx_track_video (video_id),
        INDEX idx_track_actor (assigned_actor_id),
        INDEX idx_track_match (match_actor_id),
        INDEX idx_track_clear (is_clear, status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS face_detections (
        id               INT AUTO_INCREMENT PRIMARY KEY,
        video_id         INT NOT NULL,
        track_id         INT NULL,
        frame_index      INT NOT NULL,
        timestamp_ms     INT NOT NULL,
        bbox_x           INT NOT NULL,
        bbox_y           INT NOT NULL,
        bbox_w           INT NOT NULL,
        bbox_h           INT NOT NULL,
        confidence       FLOAT NULL,
        sharpness        FLOAT NULL,
        crop_image_path  TEXT NULL,
        embedding_json   LONGTEXT NULL,
        metadata_json    LONGTEXT NULL,
        created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
        FOREIGN KEY (track_id) REFERENCES face_tracks(id) ON DELETE SET NULL,
        INDEX idx_det_video_time (video_id, timestamp_ms),
        INDEX idx_det_track (track_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS face_samples (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        actor_id        INT NOT NULL,
        source_track_id INT NULL,
        image_path      TEXT NULL,
        embedding_json  LONGTEXT NOT NULL,
        quality_score   FLOAT NULL,
        is_confirmed    BOOLEAN NOT NULL DEFAULT TRUE,
        notes           VARCHAR(255) NULL,
        created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (actor_id) REFERENCES actors(id) ON DELETE CASCADE,
        FOREIGN KEY (source_track_id) REFERENCES face_tracks(id) ON DELETE SET NULL,
        INDEX idx_sample_actor (actor_id, is_confirmed)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS overlay_events (
        id               INT AUTO_INCREMENT PRIMARY KEY,
        video_id         INT NOT NULL,
        track_id         INT NOT NULL,
        timestamp_ms     INT NOT NULL,
        bbox_x           INT NOT NULL,
        bbox_y           INT NOT NULL,
        bbox_w           INT NOT NULL,
        bbox_h           INT NOT NULL,
        label            VARCHAR(255) NULL,
        status           VARCHAR(32) NOT NULL,
        confidence       FLOAT NULL,
        created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
        FOREIGN KEY (track_id) REFERENCES face_tracks(id) ON DELETE CASCADE,
        INDEX idx_overlay_video_time (video_id, timestamp_ms)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS visual_groups (
        id                        INT AUTO_INCREMENT PRIMARY KEY,
        production_id             INT NULL,
        label                     VARCHAR(64) NOT NULL,
        review_state              ENUM('pending','confirmed','needs_split','ignored','irrelevant') NOT NULL DEFAULT 'pending',
        expansion_state           ENUM('blocked','ready','running','done') NOT NULL DEFAULT 'blocked',
        representative_image_path TEXT NULL,
        assigned_actor_id         INT NULL,
        assigned_role_id          INT NULL,
        notes                     TEXT NULL,
        created_at                TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at                TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (production_id) REFERENCES productions(id) ON DELETE SET NULL,
        FOREIGN KEY (assigned_actor_id) REFERENCES actors(id) ON DELETE SET NULL,
        FOREIGN KEY (assigned_role_id) REFERENCES roles(id) ON DELETE SET NULL,
        UNIQUE KEY uq_vg_prod_label (production_id, label),
        INDEX idx_vg_prod (production_id),
        INDEX idx_vg_review (review_state)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS visual_seeds (
        id                   INT AUTO_INCREMENT PRIMARY KEY,
        group_id             INT NULL,
        track_id             INT NULL,
        detection_id         INT NULL,
        image_path           TEXT NULL,
        embedding_json       LONGTEXT NULL,
        area_ratio           FLOAT NULL,
        sharpness            FLOAT NULL,
        confidence           FLOAT NULL,
        seed_quality_score   FLOAT NULL,
        is_removed           BOOLEAN NOT NULL DEFAULT FALSE,
        notes                VARCHAR(255) NULL,
        created_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (group_id) REFERENCES visual_groups(id) ON DELETE SET NULL,
        FOREIGN KEY (track_id) REFERENCES face_tracks(id) ON DELETE SET NULL,
        FOREIGN KEY (detection_id) REFERENCES face_detections(id) ON DELETE SET NULL,
        INDEX idx_seed_group (group_id, is_removed),
        INDEX idx_seed_track (track_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS persona_catalog (
        id                   INT AUTO_INCREMENT PRIMARY KEY,
        production_id        INT NULL,
        role_id              INT NULL,
        actor_id             INT NULL,
        voice_actor_id       INT NULL,
        voice_actor_name     VARCHAR(255) NULL,
        language             VARCHAR(32) NOT NULL DEFAULT 'de',
        relevance            TINYINT NOT NULL DEFAULT 1,
        notes                TEXT NULL,
        created_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (production_id) REFERENCES productions(id) ON DELETE CASCADE,
        FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE SET NULL,
        FOREIGN KEY (actor_id) REFERENCES actors(id) ON DELETE SET NULL,
        FOREIGN KEY (voice_actor_id) REFERENCES voice_actors(id) ON DELETE SET NULL,
        UNIQUE KEY uq_persona (production_id, role_id, language),
        INDEX idx_persona_prod (production_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS role_cast_assignments (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        production_id   INT NOT NULL,
        role_id         INT NOT NULL,
        actor_id        INT NULL,
        voice_actor_id  INT NOT NULL,
        language        VARCHAR(32) NOT NULL DEFAULT 'de',
        relevance       TINYINT NOT NULL DEFAULT 1,
        start_season    INT NOT NULL DEFAULT 1,
        start_episode   INT NOT NULL DEFAULT 1,
        notes           TEXT NULL,
        created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (production_id) REFERENCES productions(id) ON DELETE CASCADE,
        FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE,
        FOREIGN KEY (actor_id) REFERENCES actors(id) ON DELETE SET NULL,
        FOREIGN KEY (voice_actor_id) REFERENCES voice_actors(id) ON DELETE CASCADE,
        UNIQUE KEY uq_role_cast_start (production_id, role_id, language, start_season, start_episode),
        INDEX idx_role_cast_lookup (production_id, role_id, language, start_season, start_episode)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]

MIGRATION_STATEMENTS: list[str] = [
    "ALTER TABLE persona_catalog ADD COLUMN IF NOT EXISTS voice_actor_id INT NULL",
]


def _conn_params() -> dict[str, Any]:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME", "ice_nexus_db"),
    }


def get_connection() -> mariadb.Connection:
    return mariadb.connect(**_conn_params())


def _to_json(value: Any | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _from_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return fallback


def _sanitize_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        if math.isnan(value) or math.isinf(value):
            return None
    except TypeError:
        return None
    return float(value)


def _clean_optional_text(value: object, *, limit: int | None = None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if limit is not None:
        cleaned = cleaned[:limit]
    return cleaned


def _normalize_seed_workflow(metadata: dict[str, Any] | None, *, is_clear: bool) -> dict[str, Any]:
    raw_workflow = metadata.get("seed_workflow", {}) if isinstance(metadata, dict) else {}
    if not isinstance(raw_workflow, dict):
        raw_workflow = {}

    stage_default = "review" if is_clear else "seed_discovery"
    stage = str(raw_workflow.get("stage") or stage_default).strip()
    if stage not in SEED_WORKFLOW_STAGES:
        stage = stage_default

    review_state = str(raw_workflow.get("review_state") or "pending").strip()
    if review_state not in SEED_REVIEW_STATES:
        review_state = "pending"

    expansion_state = str(raw_workflow.get("expansion_state") or "blocked").strip()
    if expansion_state not in SEED_EXPANSION_STATES:
        expansion_state = "blocked"

    return {
        "mode": "seed_first",
        "stage": stage,
        "review_state": review_state,
        "group_label": _clean_optional_text(raw_workflow.get("group_label"), limit=64),
        "expansion_state": expansion_state,
        "notes": _clean_optional_text(raw_workflow.get("notes"), limit=500),
    }


def ensure_schema() -> None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        for ddl in DDL_STATEMENTS:
            cur.execute(ddl)
        for ddl in MIGRATION_STATEMENTS:
            try:
                cur.execute(ddl)
            except mariadb.Error:
                # Keep startup resilient across MariaDB minor versions.
                continue
        conn.commit()
    finally:
        conn.close()


def upsert_production_and_video(
    conn: mariadb.Connection,
    production_title: str,
    video_title: str,
    video_path: str,
    *,
    production_type: str = "series",
    season_label: str | None = None,
    episode_code: str | None = None,
    duration_ms: int | None = None,
    production_meta: dict[str, Any] | None = None,
    video_meta: dict[str, Any] | None = None,
) -> tuple[int, int]:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO productions (title, production_type, season_label, metadata_json)
        VALUES (?, ?, ?, ?)
        ON DUPLICATE KEY UPDATE
            production_type = VALUES(production_type),
            season_label = COALESCE(VALUES(season_label), season_label),
            metadata_json = VALUES(metadata_json)
        """,
        (production_title, production_type, season_label, _to_json(production_meta)),
    )
    conn.commit()
    cur.execute("SELECT id FROM productions WHERE title=?", (production_title,))
    production_id = int(cur.fetchone()[0])

    cur.execute(
        """
        INSERT INTO videos (production_id, title, episode_code, video_path, duration_ms, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?)
        ON DUPLICATE KEY UPDATE
            production_id = VALUES(production_id),
            title = VALUES(title),
            episode_code = VALUES(episode_code),
            duration_ms = VALUES(duration_ms),
            metadata_json = VALUES(metadata_json)
        """,
        (
            production_id,
            video_title,
            episode_code,
            video_path,
            duration_ms,
            _to_json(video_meta),
        ),
    )
    conn.commit()
    cur.execute("SELECT id FROM videos WHERE video_path=?", (video_path,))
    video_id = int(cur.fetchone()[0])
    return production_id, video_id


def set_video_scan_status(conn: mariadb.Connection, video_id: int, status: str) -> None:
    cur = conn.cursor()
    cur.execute("UPDATE videos SET scan_status=? WHERE id=?", (status, video_id))
    conn.commit()


def clear_video_scan_data(conn: mariadb.Connection, video_id: int) -> None:
    cur = conn.cursor()
    cur.execute("SELECT video_path, metadata_json FROM videos WHERE id=?", (video_id,))
    video_row = cur.fetchone()
    if not video_row:
        return

    video_stem = Path(str(video_row[0])).stem
    group_ids_to_recheck: set[int] = set()

    seed_params: list[Any] = [video_id, video_id, f"crops/{video_stem}/%", f"tracks/{video_stem}/%"]
    cur.execute(
        """
        SELECT DISTINCT s.id, s.group_id
        FROM visual_seeds s
        LEFT JOIN face_detections d ON d.id = s.detection_id
        LEFT JOIN face_tracks t ON t.id = s.track_id
        WHERE d.video_id = ?
           OR t.video_id = ?
           OR s.image_path LIKE ?
           OR s.image_path LIKE ?
        """,
        tuple(seed_params),
    )
    seed_rows = cur.fetchall()
    seed_ids = [int(r[0]) for r in seed_rows]
    for _, group_id in seed_rows:
        if group_id is not None:
            group_ids_to_recheck.add(int(group_id))

    if seed_ids:
        placeholders = ", ".join(["?"] * len(seed_ids))
        cur.execute(f"DELETE FROM visual_seeds WHERE id IN ({placeholders})", tuple(seed_ids))

    cur.execute("DELETE FROM overlay_events WHERE video_id=?", (video_id,))
    cur.execute("DELETE FROM face_detections WHERE video_id=?", (video_id,))
    cur.execute("DELETE FROM face_tracks WHERE video_id=?", (video_id,))

    for group_id in sorted(group_ids_to_recheck):
        cur.execute(
            "SELECT COUNT(*) FROM visual_seeds WHERE group_id=? AND is_removed=FALSE",
            (group_id,),
        )
        if int(cur.fetchone()[0] or 0) > 0:
            continue
        # Group is now empty – delete it so it doesn't clutter the UI.
        cur.execute("DELETE FROM visual_groups WHERE id=?", (group_id,))
        logger.info("Deleted empty visual_group id=%s after video rescan", group_id)

    metadata = _from_json(video_row[1], {})
    if isinstance(metadata, dict):
        metadata.pop("scan", None)
        metadata.pop("scan_stats", None)
        metadata.pop("scan_debug", None)
        metadata.pop("last_scan_result", None)
        cur.execute(
            "UPDATE videos SET metadata_json=?, last_scanned_at=NULL WHERE id=?",
            (_to_json(metadata), video_id),
        )
    else:
        cur.execute("UPDATE videos SET last_scanned_at=NULL WHERE id=?", (video_id,))

    data_root = Path(os.getenv("FACE_DATA_DIR", "data/faces")).resolve()
    for stale_dir in (data_root / "crops" / video_stem, data_root / "tracks" / video_stem):
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
            logger.info("Removed stale scan images during cleanup: %s", stale_dir)
    conn.commit()


def create_face_track(
    conn: mariadb.Connection,
    *,
    video_id: int,
    start_ms: int,
    end_ms: int,
    frame_count: int,
    mean_face_area: float | None,
    mean_sharpness: float | None,
    mean_confidence: float | None,
    stability_score: float | None,
    quality_score: float | None,
    relevance_score: float | None,
    is_clear: bool,
    status: str,
    representative_image_path: str | None,
    embedding: list[float] | None,
    metadata: dict[str, Any] | None = None,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO face_tracks (
            video_id, start_ms, end_ms, frame_count,
            mean_face_area, mean_sharpness, mean_confidence,
            stability_score, quality_score, relevance_score,
            is_clear, status, representative_image_path, embedding_json, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            video_id,
            start_ms,
            end_ms,
            frame_count,
            _sanitize_float(mean_face_area),
            _sanitize_float(mean_sharpness),
            _sanitize_float(mean_confidence),
            _sanitize_float(stability_score),
            _sanitize_float(quality_score),
            _sanitize_float(relevance_score),
            bool(is_clear),
            status,
            representative_image_path,
            _to_json(embedding),
            _to_json(metadata),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def create_face_detection(
    conn: mariadb.Connection,
    *,
    video_id: int,
    frame_index: int,
    timestamp_ms: int,
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int,
    confidence: float | None,
    sharpness: float | None,
    crop_image_path: str | None,
    embedding: list[float] | None,
    track_id: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO face_detections (
            video_id, track_id, frame_index, timestamp_ms,
            bbox_x, bbox_y, bbox_w, bbox_h,
            confidence, sharpness, crop_image_path, embedding_json, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            video_id,
            track_id,
            frame_index,
            timestamp_ms,
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h,
            _sanitize_float(confidence),
            _sanitize_float(sharpness),
            crop_image_path,
            _to_json(embedding),
            _to_json(metadata),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def assign_detection_to_track(conn: mariadb.Connection, detection_id: int, track_id: int) -> None:
    cur = conn.cursor()
    cur.execute("UPDATE face_detections SET track_id=? WHERE id=?", (track_id, detection_id))
    conn.commit()


def unlink_detection_from_track(conn: mariadb.Connection, detection_id: int) -> bool:
    """Remove a detection from its track (sets track_id = NULL).

    Returns True if a row was updated, False if detection_id did not exist.
    """
    cur = conn.cursor()
    cur.execute(
        "UPDATE face_detections SET track_id=NULL WHERE id=? AND track_id IS NOT NULL",
        (detection_id,),
    )
    conn.commit()
    return cur.rowcount > 0


def create_face_sample(
    conn: mariadb.Connection,
    *,
    actor_id: int,
    source_track_id: int | None,
    image_path: str | None,
    embedding: list[float],
    quality_score: float | None,
    is_confirmed: bool = True,
    notes: str | None = None,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO face_samples (actor_id, source_track_id, image_path, embedding_json, quality_score, is_confirmed, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            actor_id,
            source_track_id,
            image_path,
            _to_json(embedding),
            _sanitize_float(quality_score),
            bool(is_confirmed),
            notes,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def rebuild_overlay_for_video(conn: mariadb.Connection, video_id: int) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM overlay_events WHERE video_id=?", (video_id,))
    cur.execute(
        """
        SELECT
            d.track_id,
            d.timestamp_ms,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
            t.status,
            COALESCE(a1.name, a2.name, CONCAT('Track #', t.id)) AS label,
            COALESCE(t.match_score, t.quality_score)
        FROM face_detections d
        JOIN face_tracks t ON t.id = d.track_id
        LEFT JOIN actors a1 ON a1.id = t.assigned_actor_id
        LEFT JOIN actors a2 ON a2.id = t.match_actor_id
        WHERE d.video_id = ? AND d.track_id IS NOT NULL
        ORDER BY d.timestamp_ms ASC
        """,
        (video_id,),
    )
    rows = cur.fetchall()
    if not rows:
        conn.commit()
        return
    cur.executemany(
        """
        INSERT INTO overlay_events (
            video_id, track_id, timestamp_ms,
            bbox_x, bbox_y, bbox_w, bbox_h,
            label, status, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                video_id,
                int(r[0]),
                int(r[1]),
                int(r[2]),
                int(r[3]),
                int(r[4]),
                int(r[5]),
                str(r[7]),
                str(r[6]),
                _sanitize_float(r[8]),
            )
            for r in rows
        ],
    )
    conn.commit()


def list_library(conn: mariadb.Connection) -> list[dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            p.id,
            p.title,
            p.production_type,
            p.season_label,
            v.id,
            v.title,
            v.episode_code,
            v.video_path,
            v.scan_status,
            v.duration_ms,
            v.metadata_json,
            COUNT(DISTINCT t.id) AS track_count,
            SUM(CASE WHEN t.is_clear THEN 1 ELSE 0 END) AS clear_track_count
        FROM productions p
        LEFT JOIN videos v ON v.production_id = p.id
        LEFT JOIN face_tracks t ON t.video_id = v.id
        GROUP BY p.id, p.title, p.production_type, p.season_label,
                 v.id, v.title, v.episode_code, v.video_path, v.scan_status, v.duration_ms
                 , v.metadata_json
        ORDER BY p.title ASC, v.title ASC
        """
    )
    grouped: dict[int, dict[str, Any]] = {}
    for row in cur.fetchall():
        production_id = int(row[0])
        prod = grouped.setdefault(
            production_id,
            {
                "id": production_id,
                "title": row[1],
                "production_type": row[2],
                "season_label": row[3],
                "videos": [],
            },
        )
        if row[4] is None:
            continue
        metadata = _video_workflow(row[10])
        workflow = metadata.get("workflow", {})
        prod["videos"].append(
            {
                "id": int(row[4]),
                "title": row[5],
                "episode_code": row[6],
                "video_path": row[7],
                "scan_status": row[8],
                "duration_ms": row[9],
                "metadata": metadata,
                "workflow": workflow,
                "expansion_released": bool(workflow.get("expansion_released", False)),
                "track_count": int(row[11] or 0),
                "clear_track_count": int(row[12] or 0),
            }
        )
    return list(grouped.values())


def list_video_tracks(
    conn: mariadb.Connection,
    video_id: int,
    *,
    clear_only: bool = False,
    status: str | None = None,
) -> list[dict[str, Any]]:
    where = ["t.video_id = ?"]
    params: list[Any] = [video_id]
    if clear_only:
        where.append("t.is_clear = TRUE")
    if status:
        where.append("t.status = ?")
        params.append(status)

    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT
            t.id, t.video_id, t.start_ms, t.end_ms, t.frame_count,
            t.mean_face_area, t.mean_sharpness, t.mean_confidence,
            t.stability_score, t.quality_score, t.relevance_score,
            t.is_clear, t.status,
            t.assigned_actor_id, aa.name,
            t.assigned_role_id, r.name,
            t.match_actor_id, ma.name, t.match_score,
            t.representative_image_path, t.embedding_json, t.metadata_json
        FROM face_tracks t
        LEFT JOIN actors aa ON aa.id = t.assigned_actor_id
        LEFT JOIN roles r ON r.id = t.assigned_role_id
        LEFT JOIN actors ma ON ma.id = t.match_actor_id
        WHERE {' AND '.join(where)}
        ORDER BY t.start_ms ASC
        """,
        tuple(params),
    )
    rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        metadata = _from_json(row[22], {})
        if not isinstance(metadata, dict):
            metadata = {}
        seed_workflow = _normalize_seed_workflow(metadata, is_clear=bool(row[11]))
        metadata["seed_workflow"] = seed_workflow
        out.append(
            {
                "id": int(row[0]),
                "video_id": int(row[1]),
                "start_ms": int(row[2]),
                "end_ms": int(row[3]),
                "frame_count": int(row[4]),
                "mean_face_area": row[5],
                "mean_sharpness": row[6],
                "mean_confidence": row[7],
                "stability_score": row[8],
                "quality_score": row[9],
                "relevance_score": row[10],
                "is_clear": bool(row[11]),
                "status": row[12],
                "assigned_actor_id": row[13],
                "assigned_actor_name": row[14],
                "assigned_role_id": row[15],
                "assigned_role_name": row[16],
                "match_actor_id": row[17],
                "match_actor_name": row[18],
                "match_score": row[19],
                "representative_image_path": row[20],
                "embedding": _from_json(row[21], []),
                "metadata": metadata,
                "seed_workflow": seed_workflow,
            }
        )
    return out


def list_track_detections(conn: mariadb.Connection, track_id: int) -> list[dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            id, frame_index, timestamp_ms,
            bbox_x, bbox_y, bbox_w, bbox_h,
            confidence, sharpness, crop_image_path, embedding_json, metadata_json
        FROM face_detections
        WHERE track_id = ?
        ORDER BY timestamp_ms ASC
        """,
        (track_id,),
    )
    return [
        {
            "id": int(r[0]),
            "frame_index": int(r[1]),
            "timestamp_ms": int(r[2]),
            "bbox": [int(r[3]), int(r[4]), int(r[5]), int(r[6])],
            "confidence": r[7],
            "sharpness": r[8],
            "crop_image_path": r[9],
            "embedding": _from_json(r[10], []),
            "metadata": _from_json(r[11], {}),
        }
        for r in cur.fetchall()
    ]


def get_track(conn: mariadb.Connection, track_id: int) -> dict[str, Any] | None:
    cur = conn.cursor()
    cur.execute("SELECT video_id FROM face_tracks WHERE id = ?", (track_id,))
    row = cur.fetchone()
    if not row:
        return None
    video_id = int(row[0])
    tracks = list_video_tracks(conn, video_id)
    for track in tracks:
        if track["id"] == track_id:
            track["detections"] = list_track_detections(conn, track_id)
            return track
    return None


def list_overlay_events(conn: mariadb.Connection, video_id: int) -> list[dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT track_id, timestamp_ms, bbox_x, bbox_y, bbox_w, bbox_h, label, status, confidence
        FROM overlay_events
        WHERE video_id = ?
        ORDER BY timestamp_ms ASC
        """,
        (video_id,),
    )
    return [
        {
            "track_id": int(r[0]),
            "timestamp_ms": int(r[1]),
            "bbox": [int(r[2]), int(r[3]), int(r[4]), int(r[5])],
            "label": r[6],
            "status": r[7],
            "confidence": r[8],
        }
        for r in cur.fetchall()
    ]


def list_actors(conn: mariadb.Connection) -> list[dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SELECT id, name, description FROM actors ORDER BY name ASC")
    return [{"id": int(r[0]), "name": r[1], "description": r[2] or ""} for r in cur.fetchall()]


def create_actor(conn: mariadb.Connection, name: str, description: str = "") -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO actors (name, description) VALUES (?, ?) "
        "ON DUPLICATE KEY UPDATE description = VALUES(description)",
        (name.strip(), description.strip() or None),
    )
    conn.commit()
    cur.execute("SELECT id FROM actors WHERE name=?", (name.strip(),))
    return int(cur.fetchone()[0])


def list_voice_actors(conn: mariadb.Connection) -> list[dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SELECT id, name, notes FROM voice_actors ORDER BY name ASC")
    return [{"id": int(r[0]), "name": r[1], "notes": r[2] or ""} for r in cur.fetchall()]


def create_voice_actor(conn: mariadb.Connection, name: str, notes: str = "") -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO voice_actors (name, notes) VALUES (?, ?) "
        "ON DUPLICATE KEY UPDATE notes = VALUES(notes)",
        (name.strip(), _clean_optional_text(notes, limit=1000)),
    )
    conn.commit()
    cur.execute("SELECT id FROM voice_actors WHERE name=?", (name.strip(),))
    return int(cur.fetchone()[0])


def list_roles(conn: mariadb.Connection) -> list[dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SELECT id, name, description FROM roles ORDER BY name ASC")
    return [{"id": int(r[0]), "name": r[1], "description": r[2] or ""} for r in cur.fetchall()]


def create_role(conn: mariadb.Connection, name: str, description: str = "") -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO roles (name, description) VALUES (?, ?) "
        "ON DUPLICATE KEY UPDATE description = VALUES(description)",
        (name.strip(), description.strip() or None),
    )
    conn.commit()
    cur.execute("SELECT id FROM roles WHERE name=?", (name.strip(),))
    return int(cur.fetchone()[0])


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return -1.0
    if len(vec_a) != len(vec_b):
        return -1.0  # incompatible embedding dimensions (e.g. old 128-dim vs new 512-dim FaceNet)
    n = len(vec_a)
    dot = sum(float(vec_a[i]) * float(vec_b[i]) for i in range(n))
    norm_a = math.sqrt(sum(float(vec_a[i]) ** 2 for i in range(n)))
    norm_b = math.sqrt(sum(float(vec_b[i]) ** 2 for i in range(n)))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return -1.0
    return dot / (norm_a * norm_b)


def assign_track(
    conn: mariadb.Connection,
    *,
    track_id: int,
    actor_id: int,
    role_id: int | None = None,
    add_sample: bool = True,
    source: str = "manual",
) -> dict[str, Any]:
    cur = conn.cursor()
    cur.execute(
        "SELECT video_id, representative_image_path, embedding_json, quality_score FROM face_tracks WHERE id=?",
        (track_id,),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Track {track_id} not found")

    video_id = int(row[0])
    image_path = row[1]
    embedding = _from_json(row[2], [])
    quality_score = row[3]

    cur.execute(
        """
        UPDATE face_tracks
        SET assigned_actor_id=?, assigned_role_id=?, status='assigned', assignment_source=?,
            match_actor_id=?, match_score=GREATEST(COALESCE(match_score, 0), 0.99)
        WHERE id=?
        """,
        (actor_id, role_id, source, actor_id, track_id),
    )

    sample_id: int | None = None
    if add_sample and embedding:
        sample_id = create_face_sample(
            conn,
            actor_id=actor_id,
            source_track_id=track_id,
            image_path=image_path,
            embedding=[float(x) for x in embedding],
            quality_score=quality_score,
            is_confirmed=True,
            notes="auto-sample from track assignment",
        )

    if role_id is not None:
        cur.execute("SELECT production_id FROM videos WHERE id=?", (video_id,))
        prod_row = cur.fetchone()
        production_id = int(prod_row[0]) if prod_row and prod_row[0] is not None else None
        cur.execute(
            """
            INSERT INTO actor_roles (actor_id, production_id, role_id, notes)
            VALUES (?, ?, ?, ?)
            ON DUPLICATE KEY UPDATE notes = VALUES(notes)
            """,
            (actor_id, production_id, role_id, "created by track assignment"),
        )

    conn.commit()
    rebuild_overlay_for_video(conn, video_id)
    return {"track_id": track_id, "video_id": video_id, "sample_id": sample_id}


def update_track_status(conn: mariadb.Connection, track_id: int, status: str) -> None:
    allowed = {"candidate", "assigned", "ignored", "unknown", "background"}
    if status not in allowed:
        raise ValueError(f"Unsupported status: {status}")
    cur = conn.cursor()
    cur.execute("SELECT video_id FROM face_tracks WHERE id=?", (track_id,))
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Track {track_id} not found")
    video_id = int(row[0])
    cur.execute("UPDATE face_tracks SET status=? WHERE id=?", (status, track_id))
    conn.commit()
    rebuild_overlay_for_video(conn, video_id)


def update_track_seed_workflow(
    conn: mariadb.Connection,
    track_id: int,
    *,
    stage: str | None = None,
    review_state: str | None = None,
    group_label: str | None = None,
    expansion_state: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    cur = conn.cursor()
    cur.execute("SELECT video_id, is_clear, metadata_json FROM face_tracks WHERE id=?", (track_id,))
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Track {track_id} not found")

    video_id = int(row[0])
    metadata = _from_json(row[2], {})
    if not isinstance(metadata, dict):
        metadata = {}
    workflow = _normalize_seed_workflow(metadata, is_clear=bool(row[1]))

    if stage is not None:
        if stage not in SEED_WORKFLOW_STAGES:
            raise ValueError(f"Unsupported workflow stage: {stage}")
        workflow["stage"] = stage
    if review_state is not None:
        if review_state not in SEED_REVIEW_STATES:
            raise ValueError(f"Unsupported review state: {review_state}")
        workflow["review_state"] = review_state
    if expansion_state is not None:
        if expansion_state not in SEED_EXPANSION_STATES:
            raise ValueError(f"Unsupported expansion state: {expansion_state}")
        workflow["expansion_state"] = expansion_state
    if group_label is not None:
        workflow["group_label"] = _clean_optional_text(group_label, limit=64)
    if notes is not None:
        workflow["notes"] = _clean_optional_text(notes, limit=500)

    metadata["seed_workflow"] = workflow
    cur.execute("UPDATE face_tracks SET metadata_json=? WHERE id=?", (_to_json(metadata), track_id))
    conn.commit()
    return {"track_id": track_id, "video_id": video_id, "seed_workflow": workflow}


# ──────────────────────────── WP1 – visual_groups ────────────────────────────

def _get_next_group_label(conn: mariadb.Connection, production_id: int) -> str:
    """Return the next available visual_person_NNN label for a production."""
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM visual_groups WHERE production_id=?",
        (production_id,),
    )
    count = int(cur.fetchone()[0])
    return f"visual_person_{count + 1:03d}"


def create_visual_group(
    conn: mariadb.Connection,
    *,
    production_id: int,
    label: str | None = None,
    review_state: str = "pending",
    expansion_state: str = "blocked",
    representative_image_path: str | None = None,
    assigned_actor_id: int | None = None,
    assigned_role_id: int | None = None,
    notes: str | None = None,
) -> int:
    if not label:
        label = _get_next_group_label(conn, production_id)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO visual_groups
            (production_id, label, review_state, expansion_state,
             representative_image_path, assigned_actor_id, assigned_role_id, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            production_id,
            label.strip(),
            review_state,
            expansion_state,
            representative_image_path,
            assigned_actor_id,
            assigned_role_id,
            _clean_optional_text(notes, limit=500),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_visual_groups(
    conn: mariadb.Connection,
    production_id: int | None = None,
    *,
    include_seeds: bool = False,
) -> list[dict[str, Any]]:
    cur = conn.cursor()
    params: list[Any] = []
    where = "1=1"
    if production_id is not None:
        where = "g.production_id = ?"
        params.append(production_id)
    cur.execute(
        f"""
        SELECT g.id, g.production_id, p.title,
               g.label, g.review_state, g.expansion_state,
               g.representative_image_path,
               g.assigned_actor_id, aa.name,
               g.assigned_role_id, r.name,
               g.notes, g.created_at, g.updated_at,
               COUNT(DISTINCT s.id) AS seed_count,
               COUNT(DISTINCT CASE WHEN s.is_removed=FALSE THEN s.id END) AS active_seeds
        FROM visual_groups g
        LEFT JOIN productions p ON p.id = g.production_id
        LEFT JOIN actors aa ON aa.id = g.assigned_actor_id
        LEFT JOIN roles r ON r.id = g.assigned_role_id
        LEFT JOIN visual_seeds s ON s.group_id = g.id
        WHERE {where}
        GROUP BY g.id, g.production_id, p.title, g.label, g.review_state,
                 g.expansion_state, g.representative_image_path,
                 g.assigned_actor_id, aa.name, g.assigned_role_id, r.name,
                 g.notes, g.created_at, g.updated_at
        ORDER BY g.production_id ASC, g.label ASC
        """,
        tuple(params),
    )
    rows = cur.fetchall()
    out = []
    for row in rows:
        entry: dict[str, Any] = {
            "id": int(row[0]),
            "production_id": row[1],
            "production_title": row[2],
            "label": row[3],
            "review_state": row[4],
            "expansion_state": row[5],
            "representative_image_path": row[6],
            "assigned_actor_id": row[7],
            "assigned_actor_name": row[8],
            "assigned_role_id": row[9],
            "assigned_role_name": row[10],
            "notes": row[11],
            "created_at": str(row[12]),
            "updated_at": str(row[13]),
            "seed_count": int(row[14] or 0),
            "active_seeds": int(row[15] or 0),
        }
        if include_seeds:
            entry["seeds"] = list_visual_seeds(conn, group_id=entry["id"])
        out.append(entry)
    return out


def get_visual_group(conn: mariadb.Connection, group_id: int) -> dict[str, Any] | None:
    groups = list_visual_groups(conn, include_seeds=True)
    for g in groups:
        if g["id"] == group_id:
            return g
    return None


def update_visual_group(
    conn: mariadb.Connection,
    group_id: int,
    *,
    label: str | None = None,
    review_state: str | None = None,
    expansion_state: str | None = None,
    assigned_actor_id: int | None | object = _UNSET,
    assigned_role_id: int | None | object = _UNSET,
    representative_image_path: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    cur = conn.cursor()
    cur.execute("SELECT id FROM visual_groups WHERE id=?", (group_id,))
    if not cur.fetchone():
        raise ValueError(f"Visual group {group_id} not found")

    updates: list[str] = []
    params: list[Any] = []

    if label is not None:
        updates.append("label=?")
        params.append(label.strip()[:64])
    if review_state is not None:
        if review_state not in SEED_REVIEW_STATES:
            raise ValueError(f"Invalid review_state: {review_state}")
        updates.append("review_state=?")
        params.append(review_state)
    if expansion_state is not None:
        if expansion_state not in SEED_EXPANSION_STATES:
            raise ValueError(f"Invalid expansion_state: {expansion_state}")
        updates.append("expansion_state=?")
        params.append(expansion_state)
    if assigned_actor_id is not _UNSET:
        updates.append("assigned_actor_id=?")
        params.append(assigned_actor_id)
    if assigned_role_id is not _UNSET:
        updates.append("assigned_role_id=?")
        params.append(assigned_role_id)
    if representative_image_path is not None:
        updates.append("representative_image_path=?")
        params.append(representative_image_path)
    if notes is not None:
        updates.append("notes=?")
        params.append(_clean_optional_text(notes, limit=500))

    if updates:
        params.append(group_id)
        cur.execute(f"UPDATE visual_groups SET {', '.join(updates)} WHERE id=?", tuple(params))
        conn.commit()

    result = get_visual_group(conn, group_id)
    if not result:
        raise ValueError(f"Visual group {group_id} not found after update")
    return result


# ──────────────────────────── WP1 – visual_seeds ────────────────────────────

def create_visual_seed(
    conn: mariadb.Connection,
    *,
    group_id: int | None,
    track_id: int | None,
    detection_id: int | None,
    image_path: str | None,
    embedding: list[float] | None,
    area_ratio: float | None = None,
    sharpness: float | None = None,
    confidence: float | None = None,
    seed_quality_score: float | None = None,
    notes: str | None = None,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO visual_seeds
            (group_id, track_id, detection_id, image_path, embedding_json,
             area_ratio, sharpness, confidence, seed_quality_score, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            group_id,
            track_id,
            detection_id,
            image_path,
            _to_json(embedding),
            _sanitize_float(area_ratio),
            _sanitize_float(sharpness),
            _sanitize_float(confidence),
            _sanitize_float(seed_quality_score),
            notes,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_visual_seeds(
    conn: mariadb.Connection,
    *,
    group_id: int | None = None,
    include_removed: bool = False,
) -> list[dict[str, Any]]:
    cur = conn.cursor()
    where_clauses = []
    params: list[Any] = []
    if group_id is not None:
        where_clauses.append("s.group_id = ?")
        params.append(group_id)
    if not include_removed:
        where_clauses.append("s.is_removed = FALSE")
    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    cur.execute(
        f"""
        SELECT s.id, s.group_id, s.track_id, s.detection_id, s.image_path,
               s.embedding_json, s.area_ratio, s.sharpness, s.confidence, s.seed_quality_score,
               s.is_removed, s.notes, s.created_at
        FROM visual_seeds s
        {where}
        ORDER BY s.seed_quality_score DESC, s.created_at ASC
        """,
        tuple(params),
    )
    return [
        {
            "id": int(r[0]),
            "group_id": r[1],
            "track_id": r[2],
            "detection_id": r[3],
            "image_path": r[4],
            "embedding": _from_json(r[5], []),
            "area_ratio": r[6],
            "sharpness": r[7],
            "confidence": r[8],
            "seed_quality_score": r[9],
            "is_removed": bool(r[10]),
            "notes": r[11],
            "created_at": str(r[12]),
        }
        for r in cur.fetchall()
    ]


def remove_visual_seed(conn: mariadb.Connection, seed_id: int) -> bool:
    """Soft-delete a visual seed (sets is_removed=TRUE). Returns True if found."""
    cur = conn.cursor()
    cur.execute("UPDATE visual_seeds SET is_removed=TRUE WHERE id=?", (seed_id,))
    conn.commit()
    return cur.rowcount > 0


# ──────────────────────────── WP2 – conservative clustering ────────────────

def cluster_tracks_into_groups(
    conn: mariadb.Connection,
    production_id: int,
    *,
    similarity_threshold: float = 0.80,
) -> dict[str, Any]:
    """WP2: Conservative seed-first clustering.

    Only auto-merges tracks with cosine similarity >= similarity_threshold.
    Lieber Split als Fehlmerge.  Generates visual_person_NNN labels.
    Already-grouped tracks (have metadata seed_workflow.group_label) are skipped.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT t.id, t.embedding_json, t.representative_image_path, t.quality_score,
               t.metadata_json
        FROM face_tracks t
        JOIN videos v ON v.id = t.video_id
        WHERE v.production_id = ?
          AND t.is_clear = TRUE
          AND t.status NOT IN ('ignored', 'background')
        """,
        (production_id,),
    )
    raw_tracks = cur.fetchall()

    # Filter out already-grouped tracks
    track_data: list[dict[str, Any]] = []
    for row in raw_tracks:
        meta = _from_json(row[4], {})
        if isinstance(meta, dict):
            wf = meta.get("seed_workflow", {})
            if isinstance(wf, dict) and wf.get("group_label"):
                continue  # already in a group
        emb = _from_json(row[1], [])
        if not isinstance(emb, list) or not emb:
            continue
        track_data.append({
            "id": int(row[0]),
            "embedding": [float(x) for x in emb],
            "rep_path": row[2],
            "quality": float(row[3] or 0.0),
            "metadata": meta if isinstance(meta, dict) else {},
        })

    if not track_data:
        return {
            "groups_created": 0,
            "seeds_added": 0,
            "tracks_processed": 0,
            "clusters": 0,
            "skipped": len(raw_tracks),
        }

    # Greedy centroid-based clustering (conservative: only merge when sim >= threshold)
    clusters: list[list[int]] = []  # list of track_data indices
    assigned = [False] * len(track_data)

    for i in range(len(track_data)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        centroid = track_data[i]["embedding"][:]
        for j in range(i + 1, len(track_data)):
            if assigned[j]:
                continue
            sim = _cosine_similarity(centroid, track_data[j]["embedding"])
            if sim >= similarity_threshold:
                cluster.append(j)
                assigned[j] = True
                n = len(cluster)
                centroid = [
                    (centroid[k] * (n - 1) + track_data[j]["embedding"][k]) / n
                    for k in range(len(centroid))
                ]
        clusters.append(cluster)

    groups_created = 0
    seeds_added = 0

    for cluster in clusters:
        best_idx = max(cluster, key=lambda i: track_data[i]["quality"])
        best_td = track_data[best_idx]

        label = _get_next_group_label(conn, production_id)
        group_id = create_visual_group(
            conn,
            production_id=production_id,
            label=label,
            representative_image_path=best_td["rep_path"],
        )
        groups_created += 1

        for idx in cluster:
            td = track_data[idx]
            meta = td["metadata"]
            wf = meta.get("seed_workflow", {})
            if not isinstance(wf, dict):
                wf = {}
            wf["group_label"] = label
            wf.setdefault("stage", "review")
            wf.setdefault("review_state", "pending")
            wf.setdefault("expansion_state", "blocked")
            meta["seed_workflow"] = wf
            cur.execute(
                "UPDATE face_tracks SET metadata_json=? WHERE id=?",
                (_to_json(meta), td["id"]),
            )

            # For each track in cluster: reuse existing seeds or create new ones
            cur.execute(
                "SELECT id FROM visual_seeds WHERE track_id=? AND is_removed=FALSE",
                (td["id"],),
            )
            existing_seed_ids = [int(r[0]) for r in cur.fetchall()]
            if existing_seed_ids:
                # Reuse seeds already created by scanner – just assign the group
                for seed_id in existing_seed_ids:
                    cur.execute("UPDATE visual_seeds SET group_id=? WHERE id=?", (group_id, seed_id))
                seeds_added += len(existing_seed_ids)
            else:
                # No pre-existing seeds: create from top-3 detections
                cur.execute(
                    """
                    SELECT id, crop_image_path, embedding_json,
                           COALESCE(confidence, 0) AS conf,
                           COALESCE(sharpness, 0) AS sharp
                    FROM face_detections
                    WHERE track_id = ? AND crop_image_path IS NOT NULL
                    ORDER BY (COALESCE(sharpness, 0) + COALESCE(confidence, 0) * 100) DESC
                    LIMIT 3
                    """,
                    (td["id"],),
                )
                for det_row in cur.fetchall():
                    det_id = int(det_row[0])
                    crop_path = det_row[1]
                    emb = _from_json(det_row[2], [])
                    conf = _sanitize_float(float(det_row[3]))
                    sharp = _sanitize_float(float(det_row[4]))
                    q = ((conf or 0.0) * 0.5 + min((sharp or 0.0) / 300.0, 1.0) * 0.5)
                    create_visual_seed(
                        conn,
                        group_id=group_id,
                        track_id=td["id"],
                        detection_id=det_id,
                        image_path=crop_path,
                        embedding=emb,
                        sharpness=sharp,
                        confidence=conf,
                        seed_quality_score=q,
                    )
                    seeds_added += 1

    conn.commit()
    return {
        "groups_created": groups_created,
        "seeds_added": seeds_added,
        "tracks_processed": len(track_data),
        "clusters": len(clusters),
        "skipped_already_grouped": len(raw_tracks) - len(track_data),
    }


# ────────────────────────── WP4 – persona_catalog ──────────────────────────

def list_persona_catalog(
    conn: mariadb.Connection,
    production_id: int | None = None,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    where = "1=1"
    if production_id is not None:
        where = "pc.production_id = ?"
        params.append(production_id)
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT pc.id, pc.production_id, p.title,
               pc.role_id, r.name,
               pc.actor_id, a.name,
               pc.voice_actor_id, va.name,
               pc.voice_actor_name, pc.language, pc.relevance, pc.notes,
               pc.created_at, pc.updated_at
        FROM persona_catalog pc
        LEFT JOIN productions p ON p.id = pc.production_id
        LEFT JOIN roles r ON r.id = pc.role_id
        LEFT JOIN actors a ON a.id = pc.actor_id
        LEFT JOIN voice_actors va ON va.id = pc.voice_actor_id
        WHERE {where}
        ORDER BY pc.relevance DESC, r.name ASC
        """,
        tuple(params),
    )
    return [
        {
            "id": int(r[0]),
            "production_id": r[1],
            "production_title": r[2],
            "role_id": r[3],
            "role_name": r[4],
            "actor_id": r[5],
            "actor_name": r[6],
            "voice_actor_id": r[7],
            "voice_actor_name": r[8] or r[9],
            "language": r[10],
            "relevance": int(r[11]),
            "notes": r[12],
            "created_at": str(r[13]),
            "updated_at": str(r[14]),
        }
        for r in cur.fetchall()
    ]


def upsert_persona_catalog(
    conn: mariadb.Connection,
    *,
    production_id: int | None,
    role_id: int | None,
    actor_id: int | None = None,
    voice_actor_id: int | None = None,
    voice_actor_name: str | None = None,
    language: str = "de",
    relevance: int = 1,
    notes: str | None = None,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO persona_catalog
            (production_id, role_id, actor_id, voice_actor_id, voice_actor_name, language, relevance, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON DUPLICATE KEY UPDATE
            actor_id = COALESCE(VALUES(actor_id), actor_id),
            voice_actor_id = COALESCE(VALUES(voice_actor_id), voice_actor_id),
            voice_actor_name = COALESCE(VALUES(voice_actor_name), voice_actor_name),
            relevance = VALUES(relevance),
            notes = COALESCE(VALUES(notes), notes)
        """,
        (
            production_id,
            role_id,
            actor_id,
            voice_actor_id,
            _clean_optional_text(voice_actor_name, limit=255),
            language.strip()[:32] if language else "de",
            max(0, min(int(relevance), 3)),
            _clean_optional_text(notes, limit=500),
        ),
    )
    conn.commit()
    cur.execute(
        "SELECT id FROM persona_catalog WHERE production_id<=>? AND role_id<=>? AND language=?",
        (production_id, role_id, language),
    )
    row = cur.fetchone()
    return int(row[0]) if row else int(cur.lastrowid)


def delete_persona_catalog_entry(conn: mariadb.Connection, entry_id: int) -> bool:
    cur = conn.cursor()
    cur.execute("DELETE FROM persona_catalog WHERE id=?", (entry_id,))
    conn.commit()
    return cur.rowcount > 0


def list_role_cast_assignments(
    conn: mariadb.Connection,
    production_id: int | None = None,
) -> list[dict[str, Any]]:
    cur = conn.cursor()
    params: list[Any] = []
    where = "1=1"
    if production_id is not None:
        where = "rca.production_id = ?"
        params.append(production_id)
    cur.execute(
        f"""
        SELECT rca.id, rca.production_id, p.title,
               rca.role_id, r.name,
               rca.actor_id, a.name,
               rca.voice_actor_id, va.name,
               rca.language, rca.relevance, rca.start_season, rca.start_episode,
               rca.notes, rca.created_at, rca.updated_at
        FROM role_cast_assignments rca
        JOIN productions p ON p.id = rca.production_id
        JOIN roles r ON r.id = rca.role_id
        LEFT JOIN actors a ON a.id = rca.actor_id
        JOIN voice_actors va ON va.id = rca.voice_actor_id
        WHERE {where}
        ORDER BY p.title ASC, r.name ASC, rca.language ASC, rca.start_season ASC, rca.start_episode ASC
        """,
        tuple(params),
    )
    return [
        {
            "id": int(r[0]),
            "production_id": int(r[1]),
            "production_title": r[2],
            "role_id": int(r[3]),
            "role_name": r[4],
            "actor_id": r[5],
            "actor_name": r[6],
            "voice_actor_id": int(r[7]),
            "voice_actor_name": r[8],
            "language": r[9],
            "relevance": int(r[10]),
            "start_season": int(r[11]),
            "start_episode": int(r[12]),
            "notes": r[13],
            "created_at": str(r[14]),
            "updated_at": str(r[15]),
        }
        for r in cur.fetchall()
    ]


def upsert_role_cast_assignment(
    conn: mariadb.Connection,
    *,
    production_id: int,
    role_id: int,
    voice_actor_id: int,
    actor_id: int | None = None,
    language: str = "de",
    relevance: int = 1,
    start_season: int = 1,
    start_episode: int = 1,
    notes: str | None = None,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO role_cast_assignments
            (production_id, role_id, actor_id, voice_actor_id, language, relevance, start_season, start_episode, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON DUPLICATE KEY UPDATE
            actor_id = COALESCE(VALUES(actor_id), actor_id),
            voice_actor_id = VALUES(voice_actor_id),
            relevance = VALUES(relevance),
            notes = COALESCE(VALUES(notes), notes)
        """,
        (
            production_id,
            role_id,
            actor_id,
            voice_actor_id,
            (language or "de").strip()[:32],
            max(0, min(int(relevance), 3)),
            max(1, int(start_season)),
            max(1, int(start_episode)),
            _clean_optional_text(notes, limit=500),
        ),
    )
    conn.commit()
    cur.execute(
        """
        SELECT id FROM role_cast_assignments
        WHERE production_id=? AND role_id=? AND language=? AND start_season=? AND start_episode=?
        """,
        (
            production_id,
            role_id,
            (language or "de").strip()[:32],
            max(1, int(start_season)),
            max(1, int(start_episode)),
        ),
    )
    row = cur.fetchone()
    return int(row[0]) if row else int(cur.lastrowid)


def delete_role_cast_assignment(conn: mariadb.Connection, assignment_id: int) -> bool:
    cur = conn.cursor()
    cur.execute("DELETE FROM role_cast_assignments WHERE id=?", (assignment_id,))
    conn.commit()
    return cur.rowcount > 0


# ──────────────────────────── WP5 – expansion ──────────────────────────────

def trigger_group_expansion(
    conn: mariadb.Connection,
    group_id: int,
) -> dict[str, Any]:
    """WP5: Mark a confirmed group as ready for expansion.

    Only works when review_state = 'confirmed'.
    Groups with review_state in ('irrelevant', 'ignored') stay blocked.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT id, review_state, expansion_state FROM visual_groups WHERE id=?",
        (group_id,),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Visual group {group_id} not found")

    review_state = row[1]
    if review_state in ("irrelevant", "ignored"):
        return {
            "group_id": group_id,
            "ok": False,
            "reason": f"Group is {review_state} – expansion blocked to prevent noise",
            "expansion_state": row[2],
        }
    if review_state != "confirmed":
        return {
            "group_id": group_id,
            "ok": False,
            "reason": f"Group must be confirmed before expansion (current: {review_state})",
            "expansion_state": row[2],
        }

    cur.execute(
        "UPDATE visual_groups SET expansion_state='ready' WHERE id=?",
        (group_id,),
    )
    conn.commit()
    return {"group_id": group_id, "ok": True, "expansion_state": "ready"}


def block_group_expansion(
    conn: mariadb.Connection,
    group_id: int,
) -> dict[str, Any]:
    """WP5: Explicitly block expansion (e.g. after marking as irrelevant)."""
    cur = conn.cursor()
    cur.execute("SELECT id FROM visual_groups WHERE id=?", (group_id,))
    if not cur.fetchone():
        raise ValueError(f"Visual group {group_id} not found")
    cur.execute(
        "UPDATE visual_groups SET expansion_state='blocked' WHERE id=?",
        (group_id,),
    )
    conn.commit()
    return {"group_id": group_id, "ok": True, "expansion_state": "blocked"}


# ──────────────────────────── Step 1C – Expansion engine ─────────────────────

def run_expansion_for_group(
    conn: mariadb.Connection,
    group_id: int,
    *,
    match_threshold: float = 0.70,
    top_seeds: int = 10,
    allowed_video_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Step 1C: Find unassigned clear tracks in the same production that match
    a confirmed group's seed centroid and assign them to the group.

    Only works for groups with review_state='confirmed'.
    Groups marked 'irrelevant' or 'ignored' stay blocked.
    Returns a result dict with ok/tracks_matched/seeds_added.
    """
    import numpy as _np

    cur = conn.cursor()
    cur.execute(
        "SELECT id, production_id, label, review_state, expansion_state FROM visual_groups WHERE id=?",
        (group_id,),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Visual group {group_id} not found")

    review_state = str(row[3])
    if review_state in ("irrelevant", "ignored"):
        return {
            "ok": False,
            "reason": f"Group is {review_state} – expansion blocked to prevent noise",
            "tracks_matched": 0,
            "seeds_added": 0,
        }
    if review_state != "confirmed":
        return {
            "ok": False,
            "reason": f"Group must be confirmed before expansion (current: {review_state})",
            "tracks_matched": 0,
            "seeds_added": 0,
        }

    production_id = row[1]
    group_label = str(row[2])

    # Mark as running
    cur.execute("UPDATE visual_groups SET expansion_state='running' WHERE id=?", (group_id,))
    conn.commit()

    # Get seed embeddings for this group (best quality first, up to top_seeds)
    cur.execute(
        """
        SELECT embedding_json FROM visual_seeds
        WHERE group_id = ? AND is_removed = FALSE AND embedding_json IS NOT NULL
        ORDER BY seed_quality_score DESC
        LIMIT ?
        """,
        (group_id, top_seeds),
    )
    seed_embs = [_from_json(r[0], []) for r in cur.fetchall()]
    seed_embs = [e for e in seed_embs if isinstance(e, list) and len(e) > 0]

    if not seed_embs:
        cur.execute("UPDATE visual_groups SET expansion_state='blocked' WHERE id=?", (group_id,))
        conn.commit()
        return {"ok": False, "reason": "No valid seed embeddings found", "tracks_matched": 0, "seeds_added": 0}

    # Compute L2-normalised centroid of all seed embeddings
    arr = _np.array(seed_embs, dtype=_np.float64)
    centroid = arr.mean(axis=0)
    cnorm = float(_np.linalg.norm(centroid))
    if cnorm > 1e-9:
        centroid = centroid / cnorm
    centroid_list: list[float] = centroid.tolist()

    # All clear, unignored tracks in the same production.
    # Optional episode gate: only tracks from explicitly released videos.
    params: list[Any] = [production_id]
    where_video = ""
    if allowed_video_ids:
        placeholders = ", ".join(["?"] * len(allowed_video_ids))
        where_video = f" AND v.id IN ({placeholders})"
        params.extend(int(v) for v in allowed_video_ids)
    cur.execute(
        f"""
        SELECT t.id, t.embedding_json, t.representative_image_path, t.quality_score, t.metadata_json
        FROM face_tracks t
        JOIN videos v ON v.id = t.video_id
        WHERE v.production_id = ?
          AND t.is_clear = TRUE
          AND t.status NOT IN ('ignored', 'background')
          {where_video}
        """,
        tuple(params),
    )
    all_tracks = cur.fetchall()

    # Keep only tracks that have NO group_label yet (not already clustered/expanded)
    candidate_tracks = []
    for track_row in all_tracks:
        meta = _from_json(track_row[4], {})
        if isinstance(meta, dict):
            wf = meta.get("seed_workflow", {})
            if isinstance(wf, dict) and wf.get("group_label"):
                continue  # already in a group
        candidate_tracks.append(track_row)

    tracks_matched = 0
    seeds_added = 0

    for track_row in candidate_tracks:
        track_id = int(track_row[0])
        track_emb = _from_json(track_row[1], [])
        if not isinstance(track_emb, list) or not track_emb:
            continue

        sim = _cosine_similarity(centroid_list, track_emb)
        if sim < match_threshold:
            continue

        # Match – assign this track to the group by updating its seed_workflow
        meta = _from_json(track_row[4], {})
        if not isinstance(meta, dict):
            meta = {}
        wf = meta.get("seed_workflow", {})
        if not isinstance(wf, dict):
            wf = {}
        wf["group_label"] = group_label
        wf.setdefault("stage", "review")
        wf.setdefault("review_state", "pending")
        wf.setdefault("expansion_state", "blocked")
        meta["seed_workflow"] = wf
        cur.execute(
            "UPDATE face_tracks SET metadata_json=? WHERE id=?",
            (_to_json(meta), track_id),
        )

        # Create seeds from best 2 detections of matched track
        cur.execute(
            """
            SELECT id, crop_image_path, embedding_json,
                   COALESCE(confidence, 0) AS conf,
                   COALESCE(sharpness, 0) AS sharp
            FROM face_detections
            WHERE track_id = ? AND crop_image_path IS NOT NULL
            ORDER BY (COALESCE(sharpness, 0) + COALESCE(confidence, 0) * 100) DESC
            LIMIT 2
            """,
            (track_id,),
        )
        for det_row in cur.fetchall():
            det_id = int(det_row[0])
            crop_path = det_row[1]
            emb = _from_json(det_row[2], [])
            conf = _sanitize_float(float(det_row[3]))
            sharp = _sanitize_float(float(det_row[4]))
            q = ((conf or 0.0) * 0.5 + min((sharp or 0.0) / 300.0, 1.0) * 0.5)
            create_visual_seed(
                conn,
                group_id=group_id,
                track_id=track_id,
                detection_id=det_id,
                image_path=crop_path,
                embedding=emb,
                sharpness=sharp,
                confidence=conf,
                seed_quality_score=q,
            )
            seeds_added += 1

        tracks_matched += 1

    conn.commit()
    cur.execute("UPDATE visual_groups SET expansion_state='done' WHERE id=?", (group_id,))
    conn.commit()

    return {
        "ok": True,
        "group_id": group_id,
        "group_label": group_label,
        "tracks_matched": tracks_matched,
        "seeds_added": seeds_added,
        "candidates_evaluated": len(candidate_tracks),
        "expansion_state": "done",
    }


def list_face_samples(conn: mariadb.Connection, actor_id: int | None = None) -> list[dict[str, Any]]:
    where = ""
    params: tuple[Any, ...] = ()
    if actor_id is not None:
        where = "WHERE s.actor_id = ?"
        params = (actor_id,)
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT s.id, s.actor_id, a.name, s.source_track_id, s.image_path,
               s.embedding_json, s.quality_score, s.is_confirmed, s.notes, s.created_at
        FROM face_samples s
        JOIN actors a ON a.id = s.actor_id
        {where}
        ORDER BY s.created_at DESC
        """,
        params,
    )
    return [
        {
            "id": int(r[0]),
            "actor_id": int(r[1]),
            "actor_name": r[2],
            "source_track_id": r[3],
            "image_path": r[4],
            "embedding": _from_json(r[5], []),
            "quality_score": r[6],
            "is_confirmed": bool(r[7]),
            "notes": r[8],
            "created_at": str(r[9]),
        }
        for r in cur.fetchall()
    ]


def rematch_tracks(
    conn: mariadb.Connection,
    *,
    video_id: int | None = None,
    production_id: int | None = None,
    actor_id: int | None = None,
    assign_threshold: float = 0.90,
    suggest_threshold: float = 0.78,
) -> dict[str, Any]:
    cur = conn.cursor()

    sample_params: list[Any] = []
    sample_where = ["is_confirmed = TRUE"]
    if actor_id is not None:
        sample_where.append("actor_id = ?")
        sample_params.append(actor_id)
    cur.execute(
        f"SELECT actor_id, embedding_json FROM face_samples WHERE {' AND '.join(sample_where)}",
        tuple(sample_params),
    )
    samples_by_actor: dict[int, list[list[float]]] = {}
    for actor_id_raw, emb_json in cur.fetchall():
        emb = _from_json(emb_json, [])
        if not isinstance(emb, list) or not emb:
            continue
        samples_by_actor.setdefault(int(actor_id_raw), []).append([float(x) for x in emb])

    if not samples_by_actor:
        return {"updated": 0, "suggested": 0, "assigned": 0, "total": 0}

    track_where = ["1=1"]
    track_params: list[Any] = []
    if video_id is not None:
        track_where.append("video_id = ?")
        track_params.append(video_id)
    elif production_id is not None:
        track_where.append("video_id IN (SELECT id FROM videos WHERE production_id = ?)")
        track_params.append(production_id)

    cur.execute(
        f"SELECT id, video_id, embedding_json, assigned_actor_id FROM face_tracks WHERE {' AND '.join(track_where)}",
        tuple(track_params),
    )
    tracks = cur.fetchall()

    updated = 0
    suggested = 0
    assigned = 0

    for track_id_raw, _, emb_json, assigned_actor in tracks:
        emb = _from_json(emb_json, [])
        if not isinstance(emb, list) or not emb:
            continue
        vec = [float(x) for x in emb]

        best_actor: int | None = None
        best_score = -1.0
        for a_id, actor_samples in samples_by_actor.items():
            actor_best = max(_cosine_similarity(vec, s) for s in actor_samples)
            if actor_best > best_score:
                best_score = actor_best
                best_actor = a_id

        if best_actor is None:
            continue

        new_status = None
        new_assigned_actor = assigned_actor
        new_source = None
        if best_score >= assign_threshold and assigned_actor is None:
            new_assigned_actor = best_actor
            new_status = "assigned"
            new_source = "rematch"
            assigned += 1
        elif best_score >= suggest_threshold and assigned_actor is None:
            new_status = "candidate"
            suggested += 1

        cur.execute(
            """
            UPDATE face_tracks
            SET match_actor_id=?, match_score=?,
                status=COALESCE(?, status),
                assigned_actor_id=COALESCE(?, assigned_actor_id),
                assignment_source=COALESCE(?, assignment_source)
            WHERE id=?
            """,
            (
                best_actor,
                _sanitize_float(best_score),
                new_status,
                new_assigned_actor if assigned_actor is None else None,
                new_source,
                int(track_id_raw),
            ),
        )
        updated += 1

    conn.commit()

    touched_videos = {int(v[1]) for v in tracks}
    for v_id in touched_videos:
        rebuild_overlay_for_video(conn, v_id)

    return {
        "updated": updated,
        "suggested": suggested,
        "assigned": assigned,
        "total": len(tracks),
    }


def _video_workflow(metadata_json: str | None) -> dict[str, Any]:
    metadata = _from_json(metadata_json, {})
    if not isinstance(metadata, dict):
        metadata = {}
    workflow = metadata.get("workflow", {})
    if not isinstance(workflow, dict):
        workflow = {}
    workflow.setdefault("seed_scanned", False)
    workflow.setdefault("review_state", "pending")
    workflow.setdefault("expansion_released", False)
    metadata["workflow"] = workflow
    return metadata


def set_video_expansion_release(
    conn: mariadb.Connection,
    video_id: int,
    *,
    released: bool,
) -> dict[str, Any]:
    cur = conn.cursor()
    cur.execute("SELECT metadata_json FROM videos WHERE id=?", (video_id,))
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Video {video_id} not found")
    metadata = _video_workflow(row[0])
    metadata["workflow"]["expansion_released"] = bool(released)
    cur.execute("UPDATE videos SET metadata_json=? WHERE id=?", (_to_json(metadata), video_id))
    conn.commit()
    return {
        "video_id": video_id,
        "expansion_released": bool(released),
        "workflow": metadata["workflow"],
    }


def list_videos(conn: mariadb.Connection, production_id: int | None = None) -> list[dict[str, Any]]:
    cur = conn.cursor()
    if production_id is None:
        cur.execute(
            "SELECT id, production_id, title, episode_code, video_path, duration_ms, scan_status, metadata_json "
            "FROM videos ORDER BY title ASC"
        )
    else:
        cur.execute(
            "SELECT id, production_id, title, episode_code, video_path, duration_ms, scan_status, metadata_json "
            "FROM videos WHERE production_id=? ORDER BY title ASC",
            (production_id,),
        )
    out: list[dict[str, Any]] = []
    for r in cur.fetchall():
        metadata = _video_workflow(r[7])
        workflow = metadata.get("workflow", {})
        out.append(
            {
                "id": int(r[0]),
                "production_id": r[1],
                "title": r[2],
                "episode_code": r[3],
                "video_path": r[4],
                "duration_ms": r[5],
                "scan_status": r[6],
                "metadata": metadata,
                "workflow": workflow,
                "expansion_released": bool(workflow.get("expansion_released", False)),
            }
        )
    return out


def get_video(conn: mariadb.Connection, video_id: int) -> dict[str, Any] | None:
    cur = conn.cursor()
    cur.execute(
        "SELECT id, production_id, title, episode_code, video_path, duration_ms, scan_status, metadata_json "
        "FROM videos WHERE id=?",
        (video_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    metadata = _video_workflow(row[7])
    workflow = metadata.get("workflow", {})
    return {
        "id": int(row[0]),
        "production_id": row[1],
        "title": row[2],
        "episode_code": row[3],
        "video_path": row[4],
        "duration_ms": row[5],
        "scan_status": row[6],
        "metadata": metadata,
        "workflow": workflow,
        "expansion_released": bool(workflow.get("expansion_released", False)),
    }


def list_productions(conn: mariadb.Connection) -> list[dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, production_type, season_label, metadata_json FROM productions ORDER BY title ASC"
    )
    return [
        {
            "id": int(r[0]),
            "title": r[1],
            "production_type": r[2],
            "season_label": r[3],
            "metadata": _from_json(r[4], {}),
        }
        for r in cur.fetchall()
    ]
