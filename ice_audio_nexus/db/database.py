"""
ice_audio_nexus – visual-first Step-1 persistence layer.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import mariadb
from dotenv import load_dotenv

load_dotenv()

SEED_WORKFLOW_STAGES = {"seed_discovery", "review", "finished", "expansion"}
SEED_REVIEW_STATES = {"pending", "confirmed", "needs_split", "ignored", "irrelevant"}
SEED_EXPANSION_STATES = {"blocked", "ready", "running", "done"}


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
    cur.execute("DELETE FROM overlay_events WHERE video_id=?", (video_id,))
    cur.execute("DELETE FROM face_detections WHERE video_id=?", (video_id,))
    cur.execute("DELETE FROM face_tracks WHERE video_id=?", (video_id,))
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
            COUNT(DISTINCT t.id) AS track_count,
            SUM(CASE WHEN t.is_clear THEN 1 ELSE 0 END) AS clear_track_count
        FROM productions p
        LEFT JOIN videos v ON v.production_id = p.id
        LEFT JOIN face_tracks t ON t.video_id = v.id
        GROUP BY p.id, p.title, p.production_type, p.season_label,
                 v.id, v.title, v.episode_code, v.video_path, v.scan_status, v.duration_ms
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
        prod["videos"].append(
            {
                "id": int(row[4]),
                "title": row[5],
                "episode_code": row[6],
                "video_path": row[7],
                "scan_status": row[8],
                "duration_ms": row[9],
                "track_count": int(row[10] or 0),
                "clear_track_count": int(row[11] or 0),
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
    n = min(len(vec_a), len(vec_b))
    if n == 0:
        return -1.0
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


def list_videos(conn: mariadb.Connection, production_id: int | None = None) -> list[dict[str, Any]]:
    cur = conn.cursor()
    if production_id is None:
        cur.execute(
            "SELECT id, production_id, title, episode_code, video_path, duration_ms, scan_status FROM videos ORDER BY title ASC"
        )
    else:
        cur.execute(
            "SELECT id, production_id, title, episode_code, video_path, duration_ms, scan_status "
            "FROM videos WHERE production_id=? ORDER BY title ASC",
            (production_id,),
        )
    return [
        {
            "id": int(r[0]),
            "production_id": r[1],
            "title": r[2],
            "episode_code": r[3],
            "video_path": r[4],
            "duration_ms": r[5],
            "scan_status": r[6],
        }
        for r in cur.fetchall()
    ]


def get_video(conn: mariadb.Connection, video_id: int) -> dict[str, Any] | None:
    cur = conn.cursor()
    cur.execute(
        "SELECT id, production_id, title, episode_code, video_path, duration_ms, scan_status "
        "FROM videos WHERE id=?",
        (video_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return {
        "id": int(row[0]),
        "production_id": row[1],
        "title": row[2],
        "episode_code": row[3],
        "video_path": row[4],
        "duration_ms": row[5],
        "scan_status": row[6],
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
