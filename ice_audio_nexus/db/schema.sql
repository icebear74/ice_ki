-- ============================================================
-- ice_nexus_db – MariaDB 11.7 Schema
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
-- 1. voice_profiles
--    Speichert den biometrischen Stimm-Fingerabdruck.
--    VECTOR(512) = Float32-Vektoren (~2KB) – Standard für PyAnnote.
-- ============================================================
CREATE TABLE IF NOT EXISTS voice_profiles (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    voice_vector     VECTOR(512) NOT NULL COMMENT 'PyAnnote Float32-Embedding (512-dim)',
    sample_count     INT          NOT NULL DEFAULT 1 COMMENT 'Anzahl der gemittelten Samples',
    is_confirmed     BOOLEAN      NOT NULL DEFAULT FALSE COMMENT 'Durch Nutzer bestätigt?',
    created_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    VECTOR INDEX vec_idx (voice_vector) COMMENT 'MariaDB 11.7 Vektor-Index für schnelle Ähnlichkeitssuche'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- 2. identities
--    Verknüpft eine Stimme mit einem Charakter in einem
--    spezifischen Serien-/Film-Kontext.
--    Ein Synchronsprecher kann mehrere Identitäten haben.
-- ============================================================
CREATE TABLE IF NOT EXISTS identities (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    voice_id         INT          NOT NULL COMMENT 'Fremdschlüssel auf voice_profiles',
    character_name   VARCHAR(255) NOT NULL COMMENT 'z.B. Daryl Dixon',
    series_name      VARCHAR(255) NOT NULL COMMENT 'z.B. The Walking Dead',
    sync_actor_name  VARCHAR(255)          COMMENT 'Synchronsprecher (optional), z.B. Tommy Morgenstern',
    notes            TEXT                  COMMENT 'Freitext-Notizen',
    created_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (voice_id) REFERENCES voice_profiles(id) ON DELETE CASCADE,
    UNIQUE KEY uq_identity (character_name, series_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- 3. episode_segments
--    Timeline der erkannten Sprecher pro Episode.
--    Jeder Eintrag entspricht einem zusammenhängenden
--    Sprecher-Abschnitt im Video.
-- ============================================================
CREATE TABLE IF NOT EXISTS episode_segments (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    series_name      VARCHAR(255) NOT NULL COMMENT 'Serienname',
    episode_title    VARCHAR(255) NOT NULL COMMENT 'Episodentitel oder Dateiname',
    video_path       VARCHAR(512)          COMMENT 'Relativer Pfad zur Quelldatei',
    start_ms         INT          NOT NULL COMMENT 'Startzeit in Millisekunden',
    end_ms           INT          NOT NULL COMMENT 'Endzeit in Millisekunden',
    raw_speaker_id   VARCHAR(64)  NOT NULL COMMENT 'Temporäre ID der Diarization (z.B. SPEAKER_00)',
    identity_id      INT                   COMMENT 'Zugeordnete Identität (NULL = unbekannt)',
    transcript       TEXT                  COMMENT 'Whisper-Transkript des Segments',
    confidence       FLOAT                 COMMENT 'Ähnlichkeits-Score zur zugeordneten Identität (0.0–1.0)',
    is_confirmed     BOOLEAN      NOT NULL DEFAULT FALSE COMMENT 'Durch Nutzer bestätigt?',
    created_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (identity_id) REFERENCES identities(id) ON DELETE SET NULL,
    INDEX idx_episode (series_name, episode_title),
    INDEX idx_timeline (series_name, episode_title, start_ms),
    INDEX idx_speaker (raw_speaker_id),
    INDEX idx_identity (identity_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
