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
-- 1. actors
--    Biometrische Stimm-Ebene – der echte Schauspieler/Sprecher.
--    Eine Person (z.B. Patrick Stewart) besitzt mehrere
--    Identitäten (Picard, Professor X) je nach Kontext.
-- ============================================================
CREATE TABLE IF NOT EXISTS actors (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    name       VARCHAR(255) NOT NULL COMMENT 'z.B. Patrick Stewart',
    created_at TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
                                     ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_actor_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- 2. identities
--    Anker-Tabelle für eine Rolle / einen Charakter.
--    Verknüpft einen Actor mit einem Kontext-Filter
--    (z.B. actor=Patrick Stewart + context='Star Trek%' → Picard).
--    Enthält KEINEN Vektor – die Vektoren sind in voice_samples.
--    Eine Identität kann beliebig viele Vektoren besitzen
--    (Multi-Vector-Ansatz für Alterungsschutz).
-- ============================================================
CREATE TABLE IF NOT EXISTS identities (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    name           VARCHAR(255) NOT NULL COMMENT 'z.B. Jean-Luc Picard',
    description    TEXT                  COMMENT 'Optionale Beschreibung',
    actor_id       INT          DEFAULT NULL COMMENT 'Fremdschlüssel auf actors',
    context_filter VARCHAR(255) DEFAULT NULL
                   COMMENT 'SQL LIKE-Muster für Kontext-Zuordnung, z.B. Star Trek%',
    created_at     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
                                         ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_identity_name (name),
    FOREIGN KEY (actor_id) REFERENCES actors(id) ON DELETE SET NULL,
    INDEX idx_identity_actor (actor_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- 3. voice_samples
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
    updated_at   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
                                       ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (identity_id) REFERENCES identities(id) ON DELETE CASCADE,
    INDEX idx_vs_identity (identity_id),
    VECTOR INDEX vec_idx (embedding) COMMENT 'MariaDB 11.7 Vektor-Index für schnelle Ähnlichkeitssuche'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- 4. episode_segments
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
    auto_identity_id  INT          COMMENT 'Automatische Sprecherhypothese (getrennt von manueller Bestätigung)',
    matched_sample_id INT          COMMENT 'Welches voice_sample den Match auslöste',
    auto_matched_sample_id INT     COMMENT 'Welches voice_sample die Auto-Hypothese auslöste',
    match_distance    FLOAT        COMMENT 'Cosinus-Distanz (VECTOR_DISTANCE)',
    auto_match_distance FLOAT      COMMENT 'Cosinus-Distanz der Auto-Hypothese',
    match_confidence  FLOAT        COMMENT 'Abgeleitete Match-Sicherheit (0.0–1.0)',
    speaker_confidence FLOAT       COMMENT 'Abgeleitete Segment-/Sprecherqualität (0.0–1.0)',
    transcript        TEXT         COMMENT 'Whisper-Transkript des Segments',
    confidence        FLOAT        COMMENT 'Konfidenz-Score (0.0–1.0)',
    is_suggestion     BOOLEAN NOT NULL DEFAULT FALSE
                      COMMENT 'TRUE = Vorschlag, Nutzer-Bestätigung ausstehend',
    is_low_quality    BOOLEAN NOT NULL DEFAULT FALSE
                      COMMENT 'TRUE = zu kurz/rauschig/unsicher',
    is_overlap        BOOLEAN NOT NULL DEFAULT FALSE
                      COMMENT 'TRUE = überlappte bzw. problematische Sprecherbereiche',
    learning_eligible BOOLEAN NOT NULL DEFAULT FALSE
                      COMMENT 'TRUE = Segment darf für automatisches Lernen genutzt werden',
    assignment_source VARCHAR(20) NOT NULL DEFAULT 'unassigned'
                      COMMENT 'unassigned|auto|suggested|manual',
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                                         ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (identity_id)       REFERENCES identities(id)    ON DELETE SET NULL,
    FOREIGN KEY (matched_sample_id) REFERENCES voice_samples(id) ON DELETE SET NULL,
    INDEX idx_episode  (series_name, episode_title),
    INDEX idx_timeline (series_name, episode_title, start_ms),
    INDEX idx_identity (identity_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- 5. episode_probe_matches
--    Letzte Episode-interne Probe-Matching-Ergebnisse pro Segment.
-- ============================================================
CREATE TABLE IF NOT EXISTS episode_probe_matches (
    segment_id        INT PRIMARY KEY,
    probe_identity_id INT          COMMENT 'Vorgeschlagene Identität im Testlauf',
    probe_distance    FLOAT        COMMENT 'Distanz im Testlauf',
    probe_confidence  FLOAT        COMMENT 'Abgeleitete Testlauf-Sicherheit (0.0–1.0)',
    probe_status      VARCHAR(20) NOT NULL DEFAULT 'untested'
                      COMMENT 'matched|uncertain|excluded|untested',
    exclusion_reason  VARCHAR(255) COMMENT 'Warum ausgeschlossen/unsicher',
    run_label         VARCHAR(64)  COMMENT 'Freie Kennung des letzten Probe-Laufs',
    updated_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                                        ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (segment_id)        REFERENCES episode_segments(id) ON DELETE CASCADE,
    FOREIGN KEY (probe_identity_id) REFERENCES identities(id)       ON DELETE SET NULL,
    INDEX idx_probe_identity (probe_identity_id),
    INDEX idx_probe_status (probe_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
