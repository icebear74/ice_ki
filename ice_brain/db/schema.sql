-- ICE BRAIN Database Schema
-- Wird automatisch von connection.py ausgeführt wenn die Tabellen fehlen.
--
-- Mindestvoraussetzung: MariaDB 11.8 LTS (nativer VECTOR-Datentyp + HNSW-Index).
-- MySQL wird nicht mehr unterstützt.

CREATE TABLE IF NOT EXISTS users (
    user_id       VARCHAR(64)  NOT NULL PRIMARY KEY,
    username      VARCHAR(64)  NOT NULL UNIQUE,
    password_hash VARCHAR(255) NULL,          -- NULL = Erst-Login, Passwort noch nicht gesetzt
    role          ENUM('admin', 'user') NOT NULL DEFAULT 'user',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS user_memory (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id     VARCHAR(64) NOT NULL,
    category    VARCHAR(32) NOT NULL,
    content     TEXT NOT NULL,
    importance  FLOAT DEFAULT 0.5,
    is_core     BOOLEAN NOT NULL DEFAULT FALSE,
    temporal    VARCHAR(16) DEFAULT 'permanent',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    expires_at  TIMESTAMP NULL,
    enriched    BOOLEAN NOT NULL DEFAULT FALSE,
    enriched_at TIMESTAMP NULL,
    embedding   VECTOR(768) NULL,
    INDEX idx_user (user_id),
    INDEX idx_category (category),
    INDEX idx_enrichment (user_id, enriched, category),
    INDEX idx_core (user_id, is_core)
    -- VECTOR INDEX idx_mem_embedding (embedding) requires NOT NULL - add manually once embeddings are populated
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS global_memory (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    category      VARCHAR(32) NOT NULL,
    content       TEXT NOT NULL,
    source        VARCHAR(128),
    promoted_from BIGINT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS wiki_cache (
    id                BIGINT AUTO_INCREMENT PRIMARY KEY,
    title             VARCHAR(512) NOT NULL,
    query             VARCHAR(512) NOT NULL,
    summary           TEXT NOT NULL,
    full_text         MEDIUMTEXT NULL,
    keywords          TEXT NULL                    COMMENT 'Stichpunkte im Klartext die der Vektor enthält – für manuelle Pflege/Löschung',
    source_url        VARCHAR(1024),
    lang              CHAR(2) DEFAULT 'de',
    fetched_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ttl_days          INT NOT NULL DEFAULT 30,
    embedding         VECTOR(768) NULL,
    source_memory_id  BIGINT NULL                  COMMENT 'user_memory.id das diesen Abruf ausgelöst hat (erste Anreicherung)',
    UNIQUE INDEX idx_title_lang (title(200), lang),
    INDEX idx_fetched (fetched_at),
    INDEX idx_source_memory (source_memory_id),
    FULLTEXT INDEX idx_fulltext_search (title, summary, keywords)
    -- VECTOR INDEX idx_embedding (embedding) requires NOT NULL - add manually once embeddings are populated
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS memory_knowledge_link (
    memory_id    BIGINT NOT NULL,
    cache_id     BIGINT NOT NULL,
    relevance    FLOAT DEFAULT 0.5,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (memory_id, cache_id),
    FOREIGN KEY (memory_id) REFERENCES user_memory(id) ON DELETE CASCADE,
    FOREIGN KEY (cache_id) REFERENCES wiki_cache(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS wiki_chunks (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    article_id  INT NOT NULL,
    title       VARCHAR(512) NOT NULL,
    chunk_idx   SMALLINT NOT NULL,
    content     TEXT NOT NULL,
    lang        CHAR(2) DEFAULT 'de',
    embedding   VECTOR(768) NULL,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_title (title(100)),
    INDEX idx_article (article_id)
    -- VECTOR INDEX idx_wiki_embedding (embedding) requires NOT NULL - add manually once embeddings are populated
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS knowledge_entries (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    domain      VARCHAR(32) NOT NULL,
    title       VARCHAR(512),
    content     TEXT NOT NULL,
    metadata    JSON,
    source      VARCHAR(128),
    embedding   VECTOR(768) NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_domain (domain)
    -- VECTOR INDEX idx_knowledge_embedding (embedding) requires NOT NULL - add manually once embeddings are populated
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS conversation_log (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id     VARCHAR(64) NOT NULL,
    role        ENUM('user', 'assistant', 'system') NOT NULL,
    content     TEXT NOT NULL,
    model_used  VARCHAR(64),
    intent      VARCHAR(32),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_time (user_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS relations (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(64) NOT NULL,
    name            VARCHAR(128) NOT NULL,
    relation_type   VARCHAR(32) NOT NULL,
    relation_detail TEXT NULL,
    confirmed       BOOLEAN NOT NULL DEFAULT FALSE,
    enrichment_done BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user (user_id),
    INDEX idx_name (user_id, name(100)),
    INDEX idx_enrichment (user_id, enrichment_done, relation_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS relation_memory (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    relation_id BIGINT NOT NULL,
    category    VARCHAR(32) NOT NULL,
    content     TEXT NOT NULL,
    importance  FLOAT DEFAULT 0.5,
    temporal    VARCHAR(16) DEFAULT 'permanent',
    embedding   VECTOR(768) NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    expires_at  TIMESTAMP NULL,
    enriched    BOOLEAN NOT NULL DEFAULT FALSE,
    enriched_at TIMESTAMP NULL,
    FOREIGN KEY (relation_id) REFERENCES relations(id) ON DELETE CASCADE,
    INDEX idx_relation (relation_id),
    INDEX idx_category (category),
    INDEX idx_enrichment (relation_id, enriched, category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS relation_knowledge_link (
    memory_id    BIGINT NOT NULL,
    cache_id     BIGINT NOT NULL,
    relevance    FLOAT DEFAULT 0.5,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (memory_id, cache_id),
    FOREIGN KEY (memory_id) REFERENCES relation_memory(id) ON DELETE CASCADE,
    FOREIGN KEY (cache_id) REFERENCES wiki_cache(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS image_cache (
    id             BIGINT AUTO_INCREMENT PRIMARY KEY,
    source         VARCHAR(64)   NOT NULL                    COMMENT 'Herkunft: wikipedia, openstreetmap, user, …',
    source_key     VARCHAR(512)  NOT NULL                    COMMENT 'Eindeutiger Schlüssel innerhalb der Quelle (z. B. Wiki-Titel, URL)',
    mime_type      VARCHAR(64)   NOT NULL,
    image_data     MEDIUMBLOB    NOT NULL                    COMMENT 'Rohe Bilddaten (max ~16 MB)',
    thumb_data     BLOB          NULL                        COMMENT 'WebP-Vorschaubild, auto-generiert (max ~64 KB)',
    width          INT           NULL,
    height         INT           NULL,
    alt_text       VARCHAR(512)  NULL,
    original_url   VARCHAR(1024) NULL,
    fetched_at     TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    ttl_days       INT           DEFAULT 90,
    access_count   INT           DEFAULT 0,
    last_accessed  TIMESTAMP     NULL,
    UNIQUE INDEX idx_source (source, source_key(200))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS image_reference (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    image_id    BIGINT        NOT NULL,
    ref_table   VARCHAR(64)   NOT NULL                       COMMENT 'Ziel-Tabelle: user_memory, wiki_cache, …',
    ref_id      BIGINT        NOT NULL                       COMMENT 'ID in der Ziel-Tabelle',
    context     VARCHAR(128)  NULL                           COMMENT 'header, inline, thumbnail, …',
    created_at  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES image_cache(id) ON DELETE CASCADE,
    INDEX idx_ref (ref_table, ref_id),
    INDEX idx_image (image_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
