-- ice_audio_nexus visual-first Step-1 schema
-- Optional manual bootstrap (the app also auto-creates this schema on startup).

CREATE DATABASE IF NOT EXISTS ice_nexus_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE ice_nexus_db;

CREATE TABLE IF NOT EXISTS actors (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    description TEXT NULL,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_actor_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS productions (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    title           VARCHAR(255) NOT NULL,
    production_type ENUM('series','movie','other') NOT NULL DEFAULT 'series',
    season_label    VARCHAR(64) NULL,
    metadata_json   LONGTEXT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_production_title (title)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS roles (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    description TEXT NULL,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_role_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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
    UNIQUE KEY uq_video_path (video_path(512))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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
    FOREIGN KEY (match_actor_id) REFERENCES actors(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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
    FOREIGN KEY (track_id) REFERENCES face_tracks(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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
    FOREIGN KEY (source_track_id) REFERENCES face_tracks(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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
    FOREIGN KEY (track_id) REFERENCES face_tracks(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS persona_catalog (
    id                   INT AUTO_INCREMENT PRIMARY KEY,
    production_id        INT NULL,
    role_id              INT NULL,
    actor_id             INT NULL,
    voice_actor_name     VARCHAR(255) NULL,
    language             VARCHAR(32) NOT NULL DEFAULT 'de',
    relevance            TINYINT NOT NULL DEFAULT 1,
    notes                TEXT NULL,
    created_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (production_id) REFERENCES productions(id) ON DELETE CASCADE,
    FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE SET NULL,
    FOREIGN KEY (actor_id) REFERENCES actors(id) ON DELETE SET NULL,
    UNIQUE KEY uq_persona (production_id, role_id, language),
    INDEX idx_persona_prod (production_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
