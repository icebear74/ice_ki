-- ICE BRAIN Database Schema
-- Wird automatisch von connection.py ausgeführt wenn die Tabellen fehlen.
--
-- Hinweis: MySQL 8.4 (LTS) besitzt keinen nativen VECTOR-Datentyp.
-- Embeddings werden daher als MEDIUMBLOB (packed float32, 768*4 = 3072 Bytes) gespeichert.
-- Aehnlichkeitssuche erfolgt anwendungsseitig.  Upgrade auf MySQL 9.0+ ermoeglicht
-- spaeter den Wechsel zu nativem VECTOR-Typ und HNSW-Index.

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
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    expires_at  TIMESTAMP NULL,
    INDEX idx_user (user_id),
    INDEX idx_category (category)
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

CREATE TABLE IF NOT EXISTS wiki_chunks (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    article_id  INT NOT NULL,
    title       VARCHAR(512) NOT NULL,
    chunk_idx   SMALLINT NOT NULL,
    content     TEXT NOT NULL,
    lang        CHAR(2) DEFAULT 'de',
    embedding   MEDIUMBLOB NULL COMMENT 'Packed float32 embedding 768-dim (3072 Bytes)',
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_title (title(100)),
    INDEX idx_article (article_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS knowledge_entries (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    domain      VARCHAR(32) NOT NULL,
    title       VARCHAR(512),
    content     TEXT NOT NULL,
    metadata    JSON,
    source      VARCHAR(128),
    embedding   MEDIUMBLOB NULL COMMENT 'Packed float32 embedding 768-dim (3072 Bytes)',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_domain (domain)
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
