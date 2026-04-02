-- ============================================================
-- ice_nexus_db – MariaDB 11.7 Schema (Multi-Vector Identity)
-- KI-basierte Audio-Analyse & Personenidentifikation
-- ============================================================
-- Verwendung: mariadb -u root -p < db/schema.sql
-- Die Tabellenerstellung erfolgt automatisch beim ersten Start
-- via db/database.py (CREATE TABLE IF NOT EXISTS).
-- ============================================================

CREATE DATABASE IF NOT EXISTS ice_nexus_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE ice_nexus_db;

-- ============================================================
-- 1. identities
--    Anker-Tabelle für eine Person / einen Charakter.
--    Enthält KEINEN Vektor – die Vektoren sind in voice_samples.
--    Eine Identität kann beliebig viele Vektoren besitzen
--    (Multi-Vector-Ansatz für Alterungsschutz).
-- ============================================================
CREATE TABLE IF NOT EXISTS identities (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(255) NOT NULL COMMENT 'z.B. Jean-Luc Picard',
    description TEXT                  COMMENT 'Optionale Beschreibung',
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_identity_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- 2. voice_samples
--    N Vektoren pro Identität – eine Person kann viele
--    Stimmproben aus verschiedenen Kontexten haben
--    (z.B. TNG 1990, Picard-Serie 2022).
--    VECTOR(512) = Float32-Vektoren (~2KB) – Standard für PyAnnote.
-- ============================================================
CREATE TABLE IF NOT EXISTS voice_samples (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    identity_id  INT          NOT NULL COMMENT 'Fremdschlüssel auf identities',
    embedding    VECTOR(512)  NOT NULL COMMENT 'PyAnnote Float32-Embedding (512-dim)',
    context      VARCHAR(255)          COMMENT 'z.B. TNG Season 1, Picard S3E02',
    is_confirmed BOOLEAN      NOT NULL DEFAULT FALSE COMMENT 'Durch Nutzer bestätigt?',
    created_at   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (identity_id) REFERENCES identities(id) ON DELETE CASCADE,
    INDEX idx_vs_identity (identity_id),
    VECTOR INDEX vec_idx (embedding) COMMENT 'MariaDB 11.7 Vektor-Index für schnelle Ähnlichkeitssuche'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- 3. episode_segments
--    Timeline der erkannten Sprecher pro Episode.
--    Jeder Eintrag enthält das beste VECTOR_DISTANCE-Ergebnis
--    sowie die Referenz auf das auslösende voice_sample.
-- ============================================================
CREATE TABLE IF NOT EXISTS episode_segments (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    series_name       VARCHAR(255) COMMENT 'Serienname',
    episode_title     VARCHAR(255) COMMENT 'Episodentitel',
    video_path        TEXT         COMMENT 'Pfad zur Quelldatei',
    start_ms          INT NOT NULL COMMENT 'Startzeit in Millisekunden',
    end_ms            INT NOT NULL COMMENT 'Endzeit in Millisekunden',
    speaker_label     VARCHAR(100) COMMENT 'Temp. Diarization-Label (SPEAKER_01)',
    identity_id       INT          COMMENT 'Zugeordnete Identität (NULL = unbekannt)',
    matched_sample_id INT          COMMENT 'Welches voice_sample den Match auslöste',
    match_distance    FLOAT        COMMENT 'Cosinus-Distanz (VECTOR_DISTANCE)',
    transcript        TEXT         COMMENT 'Whisper-Transkript des Segments',
    confidence        FLOAT        COMMENT 'Konfidenz-Score (0.0–1.0)',
    is_suggestion     BOOLEAN NOT NULL DEFAULT FALSE
                      COMMENT 'TRUE = Vorschlag, Nutzer-Bestätigung ausstehend',
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (identity_id)       REFERENCES identities(id)    ON DELETE SET NULL,
    FOREIGN KEY (matched_sample_id) REFERENCES voice_samples(id) ON DELETE SET NULL,
    INDEX idx_episode  (series_name, episode_title),
    INDEX idx_timeline (series_name, episode_title, start_ms),
    INDEX idx_identity (identity_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
