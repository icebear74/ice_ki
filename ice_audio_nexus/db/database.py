"""
ice_audio_nexus – database.py
Auto-creates all required tables in MariaDB 11.7 on first run.

Schema (Multi-Vector Identity + Actor/Role/Production):
  actors           – real-world persons (actors and/or voice/dubbing actors)
  roles            – fictional characters / personas (e.g. 'Jean-Luc Picard')
  productions      – movies or TV series (e.g. 'Star Trek TNG')
  voice_castings   – links production + role + physical actor + voice actor + language
  identities       – voice recognition profile linked to a voice actor
  voice_samples    – n rows per identity; each holds a VECTOR(512) embedding
  episode_segments – timeline of detected speaker segments per episode
"""

import math
import os
import struct
import logging

import mariadb
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _sanitize_float(v: float | None) -> float | None:
    """Convert NaN or Inf to None so MariaDB does not raise NotSupportedError."""
    if v is None:
        return None
    try:
        if math.isnan(v) or math.isinf(v):
            return None
    except TypeError:
        return None
    return v

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = [
    # 0. Actors – real-world persons (actors and/or voice/dubbing actors)
    """
    CREATE TABLE IF NOT EXISTS actors (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        name        VARCHAR(255) NOT NULL,
        description TEXT,
        image_blob  MEDIUMBLOB NULL  COMMENT 'Profile photo stored as JPEG bytes',
        image_mime  VARCHAR(50)  NULL DEFAULT 'image/jpeg',
        created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                                       ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_actor_name (name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    # 1. Roles – fictional characters / personas
    """
    CREATE TABLE IF NOT EXISTS roles (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        name        VARCHAR(255) NOT NULL,
        description TEXT,
        image_blob  MEDIUMBLOB NULL  COMMENT 'Character image stored as JPEG bytes',
        image_mime  VARCHAR(50)  NULL DEFAULT 'image/jpeg',
        UNIQUE KEY uq_role_name (name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    # 2. Productions – movies or TV series
    """
    CREATE TABLE IF NOT EXISTS productions (
        id    INT AUTO_INCREMENT PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        year  YEAR NULL,
        type  ENUM('Movie','Series') NOT NULL DEFAULT 'Series',
        UNIQUE KEY uq_production_title (title)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    # 3. Voice castings – who plays whom in which production/language
    #    actor_id       = physical/on-screen actor (e.g. Patrick Stewart)
    #    voice_actor_id = dubbing voice actor (e.g. Rolf Schult for German dub)
    #    In the original language actor_id == voice_actor_id
    """
    CREATE TABLE IF NOT EXISTS voice_castings (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        production_id   INT NOT NULL,
        role_id         INT NOT NULL,
        actor_id        INT NOT NULL,
        voice_actor_id  INT NOT NULL,
        language        VARCHAR(10) NOT NULL DEFAULT 'de'
                        COMMENT 'BCP-47 language tag, e.g. de, en',
        FOREIGN KEY (production_id)  REFERENCES productions(id) ON DELETE CASCADE,
        FOREIGN KEY (role_id)        REFERENCES roles(id)       ON DELETE CASCADE,
        FOREIGN KEY (actor_id)       REFERENCES actors(id)      ON DELETE CASCADE,
        FOREIGN KEY (voice_actor_id) REFERENCES actors(id)      ON DELETE CASCADE,
        UNIQUE KEY uq_casting (production_id, role_id, language)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    # 4. Identity anchor – voice recognition profile (one per voice actor or character voice)
    #    NOTE: must be created BEFORE supervector_groups (which has a FK to this table)
    """
    CREATE TABLE IF NOT EXISTS identities (
        id               INT AUTO_INCREMENT PRIMARY KEY,
        name             VARCHAR(255) NOT NULL,
        description      TEXT,
        actor_id         INT DEFAULT NULL
                         COMMENT 'Legacy link to actors.id (physical actor)',
        voice_actor_id   INT DEFAULT NULL
                         COMMENT 'The voice actor whose voice this identity represents',
        voice_casting_id INT DEFAULT NULL
                         COMMENT 'Optional link to the specific voice_casting entry (role+production) this identity represents',
        context_filter   VARCHAR(255) DEFAULT NULL
                         COMMENT 'SQL LIKE pattern for context matching, e.g. Star Trek%',
        created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                                             ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_identity_name (name),
        FOREIGN KEY (actor_id)         REFERENCES actors(id)        ON DELETE SET NULL,
        FOREIGN KEY (voice_actor_id)   REFERENCES actors(id)        ON DELETE SET NULL,
        FOREIGN KEY (voice_casting_id) REFERENCES voice_castings(id) ON DELETE SET NULL,
        INDEX idx_identity_actor         (actor_id),
        INDEX idx_identity_voice_actor   (voice_actor_id),
        INDEX idx_identity_voice_casting (voice_casting_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    # 5. Supervector groups – named subsets of voice samples that form one supervector
    #    NOTE: identities (index 4) must exist first for the FK constraint
    """
    CREATE TABLE IF NOT EXISTS supervector_groups (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        identity_id INT NOT NULL,
        name        VARCHAR(255) NOT NULL COMMENT 'e.g. TNG Staffel 1-7 or Picard Serie',
        created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (identity_id) REFERENCES identities(id) ON DELETE CASCADE,
        INDEX idx_svgroup_identity (identity_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    # 6. Voice samples – many per identity, each with its own 512-dim vector
    """
    CREATE TABLE IF NOT EXISTS voice_samples (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        identity_id INT NOT NULL,
        embedding   VECTOR(512) NOT NULL,
        context     VARCHAR(255) DEFAULT NULL COMMENT 'e.g. TNG Season 1, Picard S3E02, SUPERVECTOR',
        is_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
        is_active   BOOLEAN NOT NULL DEFAULT TRUE
                    COMMENT 'FALSE = deactivated (e.g. replaced by supervector)',
        is_low_quality BOOLEAN NOT NULL DEFAULT FALSE
                    COMMENT 'True = heuristically flagged as laughter/noise/short utterance; excluded from supervectors',
        created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                                       ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (identity_id) REFERENCES identities(id) ON DELETE CASCADE,
        INDEX idx_vs_identity (identity_id),
        INDEX idx_vs_active (is_active)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    # 7. Episode segments – speaker timeline with link to matched identity
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
        is_low_quality  BOOLEAN NOT NULL DEFAULT FALSE
                        COMMENT 'True = heuristically flagged as laughter/noise/short utterance',
        tts_wav_path    TEXT NULL COMMENT 'Path to extracted TTS WAV snippet (if exported)',
        created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                                           ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (identity_id)       REFERENCES identities(id)    ON DELETE SET NULL,
        FOREIGN KEY (matched_sample_id) REFERENCES voice_samples(id) ON DELETE SET NULL,
        INDEX idx_seg_episode  (series_name, episode_title),
        INDEX idx_seg_identity (identity_id),
        INDEX idx_seg_video    (video_path(512))
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
# Vector math helpers
# ---------------------------------------------------------------------------

def normalize_vector(vec: list[float]) -> list[float]:
    """L2-normalize *vec* to unit length. Returns the original list if norm ≈ 0."""
    arr = np.array(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-6:
        return vec
    return (arr / norm).tolist()


def compute_adaptive_clusters_for_identity(
    conn: "mariadb.Connection",
    identity_id: int,
    distance_threshold: float = 0.12,
    min_samples_for_clustering: int = 8,
    outlier_ratio: float = 0.15,
) -> list[list[int]]:
    """
    Fetch all free, non-low-quality, active samples for *identity_id* and
    group them into adaptive clusters using AgglomerativeClustering.

    Algorithm:
      1. L2-normalise every vector.
      2. Remove the worst *outlier_ratio* fraction (distance to provisional mean).
      3. If fewer than *min_samples_for_clustering* samples remain → single group.
      4. Otherwise run AgglomerativeClustering (average linkage, Euclidean
         distance, no upper limit on cluster count) with *distance_threshold*.
         Samples whose pairwise distance to every existing cluster exceeds
         the threshold automatically open a new "Expert Cluster".

    Returns a list of sample-ID lists – one sublist per cluster.
    An empty list means no eligible samples exist for this identity.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, embedding FROM voice_samples
        WHERE identity_id = ?
          AND (context IS NULL OR context != 'SUPERVECTOR')
          AND used_in_group_id IS NULL
          AND is_low_quality = FALSE
          AND is_active = TRUE
        ORDER BY created_at
        """,
        (identity_id,),
    )
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return []

    valid_ids: list[int] = []
    raw_vecs: list[list[float]] = []
    for row in rows:
        vec = np.frombuffer(row[1], dtype=np.float32)
        if vec.shape[0] == 512:
            valid_ids.append(row[0])
            raw_vecs.append(normalize_vector(vec.tolist()))

    if not valid_ids:
        return []

    data = np.array(raw_vecs, dtype=np.float32)

    # Phase 1: outlier rejection
    mean_v = np.mean(data, axis=0)
    distances = np.linalg.norm(data - mean_v, axis=1)
    num_to_keep = max(1, int(len(data) * (1.0 - outlier_ratio)))
    keep_idx = np.argsort(distances)[:num_to_keep]
    filtered_data = data[keep_idx]
    filtered_ids = [valid_ids[i] for i in keep_idx]

    if len(filtered_ids) < min_samples_for_clustering:
        # Not enough samples for multi-centroid clustering → single group
        return [filtered_ids]

    # Phase 2: adaptive clustering
    try:
        from sklearn.cluster import AgglomerativeClustering  # type: ignore[import]
    except ImportError:
        logger.warning(
            "scikit-learn not available – falling back to single-cluster supervector"
        )
        return [filtered_ids]

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="euclidean",
        linkage="average",
        compute_full_tree=True,
    )
    labels = clustering.fit_predict(filtered_data)

    cluster_map: dict[int, list[int]] = {}
    for sample_id, label in zip(filtered_ids, labels):
        cluster_map.setdefault(int(label), []).append(sample_id)

    return list(cluster_map.values())


def validate_clusters(
    conn: "mariadb.Connection",
    identity_id: int,
) -> list[dict]:
    """
    Cross-validate the active supervector clusters for *identity_id*.

    For each supervector group (cluster centroid), compute what fraction of
    the source samples that were used to create it are *correctly* identified
    as closest to that centroid when compared against **all** other active
    supervector centroids for the same identity.

    Returns a list of dicts, one per group::

        {
          "group_id":              int,
          "group_name":            str,
          "sample_count":          int,
          "hit_rate_pct":          float,   # 0-100 %
          "context_distribution":  {context_str: pct_float, …},
        }
    """
    cur = conn.cursor()

    # Fetch all active supervector centroids for this identity
    cur.execute(
        """
        SELECT vs.id, vs.embedding, sg.id AS group_id, sg.name AS group_name
        FROM voice_samples vs
        JOIN supervector_groups sg ON sg.id = vs.supervector_group_id
        WHERE vs.identity_id = ?
          AND vs.context = 'SUPERVECTOR'
          AND vs.is_active = TRUE
        ORDER BY sg.id
        """,
        (identity_id,),
    )
    sv_rows = cur.fetchall()
    if not sv_rows:
        cur.close()
        return []

    sv_centroids: list[dict] = []
    for row in sv_rows:
        vec = np.frombuffer(row[1], dtype=np.float32)
        if vec.shape[0] == 512:
            sv_centroids.append({
                "sv_sample_id": row[0],
                "embedding":    np.array(normalize_vector(vec.tolist()), dtype=np.float32),
                "group_id":     row[2],
                "group_name":   row[3],
            })

    if not sv_centroids:
        cur.close()
        return []

    centroid_matrix = np.stack([c["embedding"] for c in sv_centroids], axis=0)
    group_ids_order = [c["group_id"] for c in sv_centroids]

    results: list[dict] = []
    for centroid_info in sv_centroids:
        group_id   = centroid_info["group_id"]
        group_name = centroid_info["group_name"]

        # Fetch the source samples that were merged into this group
        cur.execute(
            """
            SELECT id, embedding, context
            FROM voice_samples
            WHERE used_in_group_id = ?
            ORDER BY created_at
            """,
            (group_id,),
        )
        src_rows = cur.fetchall()

        if not src_rows:
            results.append({
                "group_id":             group_id,
                "group_name":           group_name,
                "sample_count":         0,
                "hit_rate_pct":         0.0,
                "context_distribution": {},
            })
            continue

        hits = 0
        valid_count = 0
        contexts: dict[str, int] = {}

        for src_row in src_rows:
            src_raw = np.frombuffer(src_row[1], dtype=np.float32)
            if src_raw.shape[0] != 512:
                continue
            src_emb = np.array(normalize_vector(src_raw.tolist()), dtype=np.float32)

            # Euclidean distance on L2-normalised vectors ≈ cosine distance
            dists = np.linalg.norm(centroid_matrix - src_emb, axis=1)
            nearest_group_id = group_ids_order[int(np.argmin(dists))]

            if nearest_group_id == group_id:
                hits += 1

            valid_count += 1
            ctx = src_row[2] or "unknown"
            contexts[ctx] = contexts.get(ctx, 0) + 1

        hit_rate = (hits / valid_count * 100.0) if valid_count > 0 else 0.0
        total_ctx = sum(contexts.values())
        ctx_dist = (
            {k: round(v / total_ctx * 100.0, 1) for k, v in contexts.items()}
            if total_ctx > 0
            else {}
        )

        results.append({
            "group_id":             group_id,
            "group_name":           group_name,
            "sample_count":         valid_count,
            "hit_rate_pct":         round(hit_rate, 1),
            "context_distribution": ctx_dist,
        })

    cur.close()
    return results


def calculate_robust_supervector(
    embeddings: list[list[float]],
    outlier_ratio: float = 0.2,
) -> list[float] | None:
    """
    Compute a robust centroid from *embeddings* by:
      1. L2-normalizing every vector.
      2. Computing the provisional mean.
      3. Sorting samples by Euclidean distance to the mean and
         discarding the worst *outlier_ratio* fraction.
      4. Re-computing the centroid on the remaining samples.
      5. L2-normalizing the result.

    Returns None when *embeddings* is empty.
    """
    if not embeddings:
        return None
    data = np.array([normalize_vector(e) for e in embeddings], dtype=np.float32)
    mean_v = np.mean(data, axis=0)
    distances = np.linalg.norm(data - mean_v, axis=1)
    num_to_keep = max(1, int(len(data) * (1 - outlier_ratio)))
    idx_to_keep = np.argsort(distances)[:num_to_keep]
    final_centroid = np.mean(data[idx_to_keep], axis=0)
    return normalize_vector(final_centroid.tolist())


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

        # ── Migrate pre-existing tables ──────────────────────────────────────
        # supervector_groups table (new in this version)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS supervector_groups (
                id          INT AUTO_INCREMENT PRIMARY KEY,
                identity_id INT NOT NULL,
                name        VARCHAR(255) NOT NULL,
                created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (identity_id) REFERENCES identities(id) ON DELETE CASCADE,
                INDEX idx_svgroup_identity (identity_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        # voice_samples: supervector group membership columns
        cur.execute(
            """
            ALTER TABLE voice_samples
            ADD COLUMN IF NOT EXISTS supervector_group_id INT NULL
                COMMENT 'If set, this IS the supervector sample for that group'
            """
        )
        cur.execute(
            """
            ALTER TABLE voice_samples
            ADD COLUMN IF NOT EXISTS used_in_group_id INT NULL
                COMMENT 'If set, this sample was merged into this supervector group'
            """
        )
        # actors: add new columns if missing
        for col_sql in [
            "ALTER TABLE actors ADD COLUMN IF NOT EXISTS description TEXT NULL",
            "ALTER TABLE actors ADD COLUMN IF NOT EXISTS image_blob MEDIUMBLOB NULL COMMENT 'Profile photo stored as JPEG bytes'",
            "ALTER TABLE actors ADD COLUMN IF NOT EXISTS image_mime VARCHAR(50) NULL DEFAULT 'image/jpeg'",
        ]:
            cur.execute(col_sql)

        # identities: add voice_actor_id if missing
        cur.execute(
            """
            ALTER TABLE identities
            ADD COLUMN IF NOT EXISTS voice_actor_id INT NULL
                COMMENT 'The voice actor whose voice this identity represents'
            """
        )
        # identities: add voice_casting_id if missing
        cur.execute(
            """
            ALTER TABLE identities
            ADD COLUMN IF NOT EXISTS voice_casting_id INT NULL
                COMMENT 'Optional link to the specific voice_casting entry this identity represents'
            """
        )

        # episode_segments: add embedding column if missing
        cur.execute(
            """
            ALTER TABLE episode_segments
            ADD COLUMN IF NOT EXISTS embedding VECTOR(512) NULL
                COMMENT 'Raw speaker embedding from diarization'
            """
        )
        # episode_segments: add tts_wav_path column if missing
        cur.execute(
            """
            ALTER TABLE episode_segments
            ADD COLUMN IF NOT EXISTS tts_wav_path TEXT NULL
                COMMENT 'Path to extracted TTS WAV snippet (if exported)'
            """
        )
        # episode_segments: add is_low_quality column if missing
        cur.execute(
            """
            ALTER TABLE episode_segments
            ADD COLUMN IF NOT EXISTS is_low_quality BOOLEAN NOT NULL DEFAULT FALSE
                COMMENT 'True = heuristically flagged as laughter/noise/short utterance'
            """
        )

        # voice_samples: add is_active column if missing
        cur.execute(
            """
            ALTER TABLE voice_samples
            ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE
                COMMENT 'FALSE = deactivated (e.g. replaced by supervector)'
            """
        )
        # voice_samples: ensure context column is long enough
        cur.execute(
            """
            ALTER TABLE voice_samples
            MODIFY COLUMN context VARCHAR(255) DEFAULT NULL
                COMMENT 'e.g. TNG Season 1, Picard S3E02, SUPERVECTOR'
            """
        )
        # voice_samples: add is_low_quality column if missing
        cur.execute(
            """
            ALTER TABLE voice_samples
            ADD COLUMN IF NOT EXISTS is_low_quality BOOLEAN NOT NULL DEFAULT FALSE
                COMMENT 'True = heuristically flagged as laughter/noise/short utterance; excluded from supervectors'
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
               COUNT(vs.id) AS sample_count,
               i.voice_actor_id,
               a.name AS voice_actor_name,
               i.voice_casting_id,
               vc.language AS casting_language,
               r.name  AS casting_role_name,
               p.title AS casting_production_title
        FROM identities i
        LEFT JOIN voice_samples vs ON vs.identity_id = i.id
        LEFT JOIN actors a ON a.id = i.voice_actor_id
        LEFT JOIN voice_castings vc ON vc.id = i.voice_casting_id
        LEFT JOIN roles       r ON r.id  = vc.role_id
        LEFT JOIN productions p ON p.id  = vc.production_id
        GROUP BY i.id
        ORDER BY i.name
    """)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_identity(conn: mariadb.Connection, identity_id: int) -> dict | None:
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, description, voice_actor_id, voice_casting_id "
        "FROM identities WHERE id = ?",
        (identity_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {
        "id": row[0], "name": row[1], "description": row[2],
        "voice_actor_id": row[3], "voice_casting_id": row[4],
    }


def create_identity(conn: mariadb.Connection, name: str, description: str = "",
                    voice_actor_id: int | None = None,
                    voice_casting_id: int | None = None) -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO identities (name, description, voice_actor_id, voice_casting_id) "
        "VALUES (?, ?, ?, ?)",
        (name, description, voice_actor_id, voice_casting_id),
    )
    conn.commit()
    return cur.lastrowid


def update_identity(conn: mariadb.Connection, identity_id: int, name: str, description: str,
                    voice_actor_id: int | None = None,
                    voice_casting_id: int | None = None) -> None:
    cur = conn.cursor()
    cur.execute(
        "UPDATE identities SET name = ?, description = ?, voice_actor_id = ?, "
        "voice_casting_id = ? WHERE id = ?",
        (name, description, voice_actor_id, voice_casting_id, identity_id),
    )
    conn.commit()


def delete_identity(conn: mariadb.Connection, identity_id: int) -> None:
    """Delete an identity and all its voice samples (CASCADE handles samples)."""
    cur = conn.cursor()
    cur.execute("DELETE FROM identities WHERE id = ?", (identity_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Supervector management
# ---------------------------------------------------------------------------

def refresh_supervectors(conn: mariadb.Connection) -> dict:
    """
    Adaptive multi-centroid supervector mode: for every identity, delete ALL
    existing supervector groups (full revert), then re-cluster the free,
    non-low-quality samples using distance-based AgglomerativeClustering.

    • If an identity has fewer than 8 eligible samples a single robust
      supervector is created (existing behaviour).
    • Otherwise an unlimited number of clusters is formed automatically –
      every group of samples whose intra-cluster distance exceeds the
      threshold (default 0.12) opens a new "Expert Cluster".

    Returns a summary dict::

        {identity_name: {"samples": int, "clusters": int}, …}
    """
    from datetime import date
    today = date.today()

    cur = conn.cursor()
    summary: dict = {}
    try:
        cur.execute("SELECT id, name FROM identities ORDER BY name")
        identities_list = cur.fetchall()
    finally:
        cur.close()

    for identity_id, identity_name in identities_list:
        try:
            # 1. Revert all existing supervectors so samples are free again
            revert_supervectors(conn, identity_id)

            # 2. Compute adaptive clusters
            clusters = compute_adaptive_clusters_for_identity(conn, identity_id)
            if not clusters:
                continue

            total_samples = 0
            num_clusters  = len(clusters)
            for i, cluster_sample_ids in enumerate(clusters, 1):
                if num_clusters > 1:
                    cluster_name = f"Auto {today} – Cluster {i}/{num_clusters}"
                else:
                    cluster_name = f"Auto {today}"
                try:
                    create_named_supervector(
                        conn, identity_id, cluster_name, cluster_sample_ids
                    )
                    total_samples += len(cluster_sample_ids)
                except Exception as exc:
                    logger.warning(
                        "Auto-supervector skipped for %s cluster %d: %s",
                        identity_name, i, exc,
                    )

            if total_samples > 0:
                summary[identity_name] = {
                    "samples":  total_samples,
                    "clusters": num_clusters,
                }
        except Exception as exc:
            logger.warning(
                "refresh_supervectors: identity '%s' failed: %s",
                identity_name, exc,
            )

    return summary


def revert_supervectors(conn: mariadb.Connection, identity_id: int) -> int:
    """
    Revert ALL supervector groups for *identity_id*: delete every supervector
    sample and reactivate all original source samples.

    Returns the total number of reactivated samples.
    """
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id FROM supervector_groups WHERE identity_id = ?",
            (identity_id,),
        )
        group_ids = [row[0] for row in cur.fetchall()]
    finally:
        cur.close()

    total = 0
    for gid in group_ids:
        total += revert_supervector_group(conn, gid)
    return total


def get_identity_vector_stats(
    conn: mariadb.Connection,
    identity_id: int,
) -> dict:
    """
    Compute variance and per-sample distance-to-centroid for the active,
    non-supervector, non-low-quality samples of *identity_id*.

    Returns:
        {
          avg_distance:     float  – mean Euclidean distance to the centroid
                                     (on L2-normalized vectors ≈ cosine metric),
          variance:         float  – variance of those distances,
          sample_distances: list[{id: int, distance: float}]
        }
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, embedding
        FROM voice_samples
        WHERE identity_id = ?
          AND is_active = TRUE
          AND (context IS NULL OR context != 'SUPERVECTOR')
          AND is_low_quality = FALSE
        ORDER BY created_at
        """,
        (identity_id,),
    )
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return {"avg_distance": 0.0, "variance": 0.0, "sample_distances": []}

    ids: list[int] = []
    vecs: list[list[float]] = []
    for row in rows:
        vec = np.frombuffer(row[1], dtype=np.float32)
        if vec.shape[0] == 512:
            ids.append(row[0])
            vecs.append(normalize_vector(vec.tolist()))

    if not vecs:
        return {"avg_distance": 0.0, "variance": 0.0, "sample_distances": []}

    data = np.array(vecs, dtype=np.float32)
    centroid = np.mean(data, axis=0)
    dists = np.linalg.norm(data - centroid, axis=1)

    sample_distances = [
        {"id": sid, "distance": float(d)}
        for sid, d in zip(ids, dists)
    ]
    return {
        "avg_distance": float(np.mean(dists)),
        "variance": float(np.var(dists)),
        "sample_distances": sample_distances,
    }


# ---------------------------------------------------------------------------
# Actor CRUD
# ---------------------------------------------------------------------------

def list_actors(conn: mariadb.Connection, production_id: int | None = None) -> list[dict]:
    cur = conn.cursor()
    if production_id is not None:
        cur.execute(
            """
            SELECT a.id, a.name, a.description,
                   (a.image_blob IS NOT NULL) AS has_image,
                   COUNT(DISTINCT i.id) AS identity_count,
                   (COUNT(DISTINCT vc.id) > 0) AS in_production
            FROM actors a
            LEFT JOIN identities i ON i.voice_actor_id = a.id
            LEFT JOIN voice_castings vc
                   ON (vc.actor_id = a.id OR vc.voice_actor_id = a.id)
                  AND vc.production_id = ?
            GROUP BY a.id
            ORDER BY in_production DESC, a.name
            """,
            (production_id,),
        )
    else:
        cur.execute("""
            SELECT a.id, a.name, a.description,
                   (a.image_blob IS NOT NULL) AS has_image,
                   COUNT(i.id) AS identity_count
            FROM actors a
            LEFT JOIN identities i ON i.voice_actor_id = a.id
            GROUP BY a.id
            ORDER BY a.name
        """)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_actor(conn: mariadb.Connection, actor_id: int) -> dict | None:
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, description, (image_blob IS NOT NULL) AS has_image, created_at, updated_at "
        "FROM actors WHERE id = ?",
        (actor_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def create_actor(conn: mariadb.Connection, name: str, description: str = "") -> int:
    cur = conn.cursor()
    cur.execute("INSERT INTO actors (name, description) VALUES (?, ?)", (name, description))
    conn.commit()
    return cur.lastrowid


def update_actor(conn: mariadb.Connection, actor_id: int, name: str, description: str = "") -> None:
    cur = conn.cursor()
    cur.execute("UPDATE actors SET name = ?, description = ? WHERE id = ?", (name, description, actor_id))
    conn.commit()


def update_actor_image(conn: mariadb.Connection, actor_id: int, image_bytes: bytes, mime: str = "image/jpeg") -> None:
    cur = conn.cursor()
    cur.execute(
        "UPDATE actors SET image_blob = ?, image_mime = ? WHERE id = ?",
        (image_bytes, mime, actor_id),
    )
    conn.commit()


def get_actor_image(conn: mariadb.Connection, actor_id: int) -> tuple[bytes, str] | None:
    cur = conn.cursor()
    cur.execute("SELECT image_blob, image_mime FROM actors WHERE id = ?", (actor_id,))
    row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    return (bytes(row[0]), row[1] or "image/jpeg")


def delete_actor(conn: mariadb.Connection, actor_id: int) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM actors WHERE id = ?", (actor_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Role CRUD
# ---------------------------------------------------------------------------

def list_roles(conn: mariadb.Connection) -> list[dict]:
    cur = conn.cursor()
    cur.execute("""
        SELECT id, name, description, (image_blob IS NOT NULL) AS has_image
        FROM roles ORDER BY name
    """)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_role(conn: mariadb.Connection, role_id: int) -> dict | None:
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, description, (image_blob IS NOT NULL) AS has_image FROM roles WHERE id = ?",
        (role_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def create_role(conn: mariadb.Connection, name: str, description: str = "") -> int:
    cur = conn.cursor()
    cur.execute("INSERT INTO roles (name, description) VALUES (?, ?)", (name, description))
    conn.commit()
    return cur.lastrowid


def update_role(conn: mariadb.Connection, role_id: int, name: str, description: str = "") -> None:
    cur = conn.cursor()
    cur.execute("UPDATE roles SET name = ?, description = ? WHERE id = ?", (name, description, role_id))
    conn.commit()


def update_role_image(conn: mariadb.Connection, role_id: int, image_bytes: bytes, mime: str = "image/jpeg") -> None:
    cur = conn.cursor()
    cur.execute(
        "UPDATE roles SET image_blob = ?, image_mime = ? WHERE id = ?",
        (image_bytes, mime, role_id),
    )
    conn.commit()


def get_role_image(conn: mariadb.Connection, role_id: int) -> tuple[bytes, str] | None:
    cur = conn.cursor()
    cur.execute("SELECT image_blob, image_mime FROM roles WHERE id = ?", (role_id,))
    row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    return (bytes(row[0]), row[1] or "image/jpeg")


def delete_role(conn: mariadb.Connection, role_id: int) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM roles WHERE id = ?", (role_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Production CRUD
# ---------------------------------------------------------------------------

def list_productions(conn: mariadb.Connection) -> list[dict]:
    cur = conn.cursor()
    cur.execute("SELECT id, title, year, type FROM productions ORDER BY title")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_production(conn: mariadb.Connection, production_id: int) -> dict | None:
    cur = conn.cursor()
    cur.execute("SELECT id, title, year, type FROM productions WHERE id = ?", (production_id,))
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def create_production(conn: mariadb.Connection, title: str, year: int | None = None,
                      prod_type: str = "Series") -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO productions (title, year, type) VALUES (?, ?, ?)",
        (title, year, prod_type),
    )
    conn.commit()
    return cur.lastrowid


def update_production(conn: mariadb.Connection, production_id: int, title: str,
                      year: int | None = None, prod_type: str = "Series") -> None:
    cur = conn.cursor()
    cur.execute(
        "UPDATE productions SET title = ?, year = ?, type = ? WHERE id = ?",
        (title, year, prod_type, production_id),
    )
    conn.commit()


def delete_production(conn: mariadb.Connection, production_id: int) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM productions WHERE id = ?", (production_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Voice casting CRUD
# ---------------------------------------------------------------------------

def list_voice_castings(conn: mariadb.Connection, production_id: int | None = None) -> list[dict]:
    cur = conn.cursor()
    if production_id is not None:
        cur.execute(
            """
            SELECT vc.id, vc.production_id, p.title AS production_title,
                   vc.role_id, r.name AS role_name,
                   vc.actor_id, a.name AS actor_name,
                   vc.voice_actor_id, va.name AS voice_actor_name,
                   vc.language
            FROM voice_castings vc
            JOIN productions p  ON p.id  = vc.production_id
            JOIN roles        r ON r.id  = vc.role_id
            JOIN actors       a ON a.id  = vc.actor_id
            JOIN actors       va ON va.id = vc.voice_actor_id
            WHERE vc.production_id = ?
            ORDER BY r.name, vc.language
            """,
            (production_id,),
        )
    else:
        cur.execute(
            """
            SELECT vc.id, vc.production_id, p.title AS production_title,
                   vc.role_id, r.name AS role_name,
                   vc.actor_id, a.name AS actor_name,
                   vc.voice_actor_id, va.name AS voice_actor_name,
                   vc.language
            FROM voice_castings vc
            JOIN productions p  ON p.id  = vc.production_id
            JOIN roles        r ON r.id  = vc.role_id
            JOIN actors       a ON a.id  = vc.actor_id
            JOIN actors       va ON va.id = vc.voice_actor_id
            ORDER BY p.title, r.name, vc.language
            """
        )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def create_voice_casting(conn: mariadb.Connection, production_id: int, role_id: int,
                         actor_id: int, voice_actor_id: int, language: str = "de") -> int:
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO voice_castings (production_id, role_id, actor_id, voice_actor_id, language)
           VALUES (?, ?, ?, ?, ?)""",
        (production_id, role_id, actor_id, voice_actor_id, language),
    )
    conn.commit()
    return cur.lastrowid


def delete_voice_casting(conn: mariadb.Connection, casting_id: int) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM voice_castings WHERE id = ?", (casting_id,))
    conn.commit()


def update_voice_casting(conn: mariadb.Connection, casting_id: int, production_id: int,
                         role_id: int, actor_id: int, voice_actor_id: int,
                         language: str = "de") -> None:
    cur = conn.cursor()
    cur.execute(
        """UPDATE voice_castings
           SET production_id = ?, role_id = ?, actor_id = ?, voice_actor_id = ?, language = ?
           WHERE id = ?""",
        (production_id, role_id, actor_id, voice_actor_id, language, casting_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Named supervector groups
# ---------------------------------------------------------------------------

def list_supervector_groups(conn: mariadb.Connection, identity_id: int) -> list[dict]:
    """Return all supervector groups for *identity_id* with their sample counts."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT sg.id, sg.name, sg.created_at,
               COUNT(sv.id) AS sample_count
        FROM supervector_groups sg
        LEFT JOIN voice_samples sv ON sv.used_in_group_id = sg.id
        WHERE sg.identity_id = ?
        GROUP BY sg.id
        ORDER BY sg.created_at
        """,
        (identity_id,),
    )
    results = []
    for row in cur.fetchall():
        results.append({
            "id":           row[0],
            "name":         row[1],
            "created_at":   str(row[2]) if row[2] is not None else None,
            "sample_count": row[3],
        })
    return results


def list_free_samples(conn: mariadb.Connection, identity_id: int) -> list[dict]:
    """Return active raw samples not yet merged into any supervector group."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, context, is_confirmed, is_active, is_low_quality, created_at
        FROM voice_samples
        WHERE identity_id = ?
          AND (context IS NULL OR context != 'SUPERVECTOR')
          AND used_in_group_id IS NULL
        ORDER BY created_at
        """,
        (identity_id,),
    )
    results = []
    for row in cur.fetchall():
        results.append({
            "id":           row[0],
            "context":      row[1],
            "is_confirmed": bool(row[2]),
            "is_active":    bool(row[3]),
            "is_low_quality": bool(row[4]),
            "created_at":   str(row[5]) if row[5] is not None else None,
        })
    return results


def list_group_samples(conn: mariadb.Connection, group_id: int) -> list[dict]:
    """Return the source samples that were merged into *group_id*."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, context, is_confirmed, is_low_quality, created_at
        FROM voice_samples
        WHERE used_in_group_id = ?
        ORDER BY created_at
        """,
        (group_id,),
    )
    results = []
    for row in cur.fetchall():
        results.append({
            "id":             row[0],
            "context":        row[1],
            "is_confirmed":   bool(row[2]),
            "is_low_quality": bool(row[3]),
            "created_at":     str(row[4]) if row[4] is not None else None,
        })
    return results


def get_group_vector_stats(conn: mariadb.Connection, group_id: int) -> dict:
    """
    Compute per-sample distance-to-centroid for the source samples of *group_id*.

    Returns the same shape as get_identity_vector_stats():
        {avg_distance: float, variance: float, sample_distances: [{id, distance}, …]}
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, embedding
        FROM voice_samples
        WHERE used_in_group_id = ?
        ORDER BY created_at
        """,
        (group_id,),
    )
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return {"avg_distance": 0.0, "variance": 0.0, "sample_distances": []}

    ids: list[int] = []
    vecs: list[list[float]] = []
    for row in rows:
        vec = np.frombuffer(row[1], dtype=np.float32)
        if vec.shape[0] == 512:
            ids.append(row[0])
            vecs.append(normalize_vector(vec.tolist()))

    if not vecs:
        return {"avg_distance": 0.0, "variance": 0.0, "sample_distances": []}

    data = np.array(vecs, dtype=np.float32)
    centroid = np.mean(data, axis=0)
    dists = np.linalg.norm(data - centroid, axis=1)

    sample_distances = [
        {"id": sid, "distance": float(d)}
        for sid, d in zip(ids, dists)
    ]
    return {
        "avg_distance": float(np.mean(dists)),
        "variance": float(np.var(dists)),
        "sample_distances": sample_distances,
    }


def create_named_supervector(
    conn: mariadb.Connection,
    identity_id: int,
    name: str,
    sample_ids: list[int],
) -> int:
    """
    Compute the centroid of the selected *sample_ids*, store it as a new
    SUPERVECTOR voice_sample linked to a new supervector_group, and mark all
    source samples as inactive / merged.

    Returns the new supervector_group.id.
    Raises ValueError if any sample_id is invalid (wrong identity, already merged,
    is itself a supervector, or is missing).
    """
    if not sample_ids:
        raise ValueError("sample_ids must not be empty")

    cur = conn.cursor()

    # Build a parameterised IN-clause from the validated integer list.
    # {fmt} expands to "?,?,?,...,?" – only placeholders, never user data –
    # so there is no SQL-injection risk despite the f-string.  The actual
    # values are passed as bound parameters in the second argument.
    fmt = ",".join("?" for _ in sample_ids)
    cur.execute(
        f"""
        SELECT id FROM voice_samples
        WHERE id IN ({fmt})
          AND identity_id = ?
          AND (context IS NULL OR context != 'SUPERVECTOR')
          AND used_in_group_id IS NULL
          AND is_active = TRUE
        """,  # noqa: S608
        (*sample_ids, identity_id),
    )
    valid_ids = {row[0] for row in cur.fetchall()}
    invalid = set(sample_ids) - valid_ids
    if invalid:
        raise ValueError(f"Invalid or already-merged sample IDs: {invalid}")

    # Load embeddings
    cur.execute(
        f"SELECT id, embedding FROM voice_samples WHERE id IN ({fmt})",  # noqa: S608
        sample_ids,
    )
    embeddings = []
    for row in cur.fetchall():
        vec = np.frombuffer(row[1], dtype=np.float32)
        if vec.shape[0] == 512:
            embeddings.append(vec)

    if not embeddings:
        raise ValueError("No valid 512-dim embeddings found in selected samples")

    robust = calculate_robust_supervector([v.tolist() for v in embeddings])
    if robust is None:
        raise ValueError("Robust supervector calculation returned no result")
    supervector = np.array(robust, dtype=np.float32)

    try:
        # Create the group record
        cur.execute(
            "INSERT INTO supervector_groups (identity_id, name) VALUES (?, ?)",
            (identity_id, name),
        )
        group_id = cur.lastrowid

        # Insert the supervector sample
        cur.execute(
            """INSERT INTO voice_samples
                   (identity_id, embedding, context, is_confirmed, is_active,
                    supervector_group_id)
               VALUES (?, ?, 'SUPERVECTOR', TRUE, TRUE, ?)
            """,
            (identity_id, vector_to_bytes(supervector.tolist()), group_id),
        )

        # Mark source samples as merged (inactive, linked to group)
        cur.execute(
            f"""UPDATE voice_samples
               SET is_active = FALSE, used_in_group_id = ?
               WHERE id IN ({fmt})""",  # noqa: S608
            (group_id, *sample_ids),
        )

        conn.commit()
        return group_id
    except Exception:
        conn.rollback()
        raise


def revert_supervector_group(conn: mariadb.Connection, group_id: int) -> int:
    """
    Delete the supervector for *group_id* and reactivate all original samples
    that were merged into it.  Deletes the group record afterwards.

    Returns the number of reactivated samples.
    """
    cur = conn.cursor()
    try:
        # Delete the supervector sample linked to this group
        cur.execute(
            "DELETE FROM voice_samples WHERE supervector_group_id = ?",
            (group_id,),
        )
        # Reactivate merged source samples
        cur.execute(
            """UPDATE voice_samples
               SET is_active = TRUE, used_in_group_id = NULL
               WHERE used_in_group_id = ?""",
            (group_id,),
        )
        reactivated = cur.rowcount
        # Delete the group record
        cur.execute("DELETE FROM supervector_groups WHERE id = ?", (group_id,))
        conn.commit()
        return reactivated
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


# ---------------------------------------------------------------------------
# Voice sample CRUD
# ---------------------------------------------------------------------------

def add_voice_sample(
    conn: mariadb.Connection,
    identity_id: int,
    embedding: list[float],
    context: str = "",
    is_confirmed: bool = False,
    is_low_quality: bool = False,
) -> int:
    # Replace NaN / ±inf (e.g. from old scanner data) with 0.0 so MariaDB VECTOR
    # does not reject the row with "Incorrect vector value".
    embedding = [v if math.isfinite(v) else 0.0 for v in embedding]
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO voice_samples (identity_id, embedding, context, is_confirmed, is_low_quality)
           VALUES (?, ?, ?, ?, ?)""",
        (identity_id, vector_to_bytes(embedding), context or None, is_confirmed, is_low_quality),
    )
    conn.commit()
    return cur.lastrowid


def list_voice_samples(conn: mariadb.Connection, identity_id: int) -> list[dict]:
    cur = conn.cursor()
    cur.execute(
        """SELECT id, identity_id, embedding, context, is_confirmed, is_active,
                  is_low_quality, created_at
           FROM voice_samples WHERE identity_id = ? ORDER BY created_at""",
        (identity_id,),
    )
    results = []
    for row in cur.fetchall():
        results.append({
            "id":             row[0],
            "identity_id":    row[1],
            "embedding":      bytes_to_vector(row[2]),
            "context":        row[3],
            "is_confirmed":   bool(row[4]),
            "is_active":      bool(row[5]),
            "is_low_quality": bool(row[6]),
            "created_at":     str(row[7]),
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
    min_margin: float = 0.07,
) -> dict:
    """
    Search all voice_samples using VEC_DISTANCE_COSINE (MariaDB 11.7+) and
    return the closest match.

    *min_margin* guards against cross-identity confusion (e.g. two characters
    whose embeddings land close together).  Even if the best match is within
    *match_threshold*, the result is only considered a confirmed match when the
    next-closest sample from a **different** identity is at least *min_margin*
    further away.  If the margin is too small the result is downgraded:
      • confirmed → suggest  (if still within suggest_threshold + margin)
      • suggest   → unknown

    Returns a dict with keys:
      status        – 'matched' | 'suggest' | 'unknown'
      identity_id   – int or None
      identity_name – str or None
      sample_id     – int or None   (which sample triggered the match)
      sample_context– str or None
      distance      – float or None
      second_distance – float or None  (runner-up distance, useful for debugging)
    """
    vec_bytes = vector_to_bytes(embedding)
    cur = conn.cursor()
    # Fetch the two closest samples (potentially from different identities)
    # Only consider active samples (is_active = TRUE).
    cur.execute(
        """
        SELECT vs.id,
               vs.identity_id,
               i.name,
               vs.context,
               VEC_DISTANCE_COSINE(vs.embedding, ?) AS dist
        FROM voice_samples vs
        JOIN identities i ON i.id = vs.identity_id
        WHERE vs.is_active = TRUE
        ORDER BY dist ASC
        LIMIT 10
        """,
        (vec_bytes,),
    )
    rows = cur.fetchall()
    if not rows:
        return {"status": "unknown", "identity_id": None, "identity_name": None,
                "sample_id": None, "sample_context": None, "distance": None,
                "second_distance": None}

    # VEC_DISTANCE_COSINE returns NULL when the stored embedding is malformed.
    # Filter those rows out so float() never receives None.
    rows = [r for r in rows if r[4] is not None]
    if not rows:
        return {"status": "unknown", "identity_id": None, "identity_name": None,
                "sample_id": None, "sample_context": None, "distance": None,
                "second_distance": None}

    sample_id, identity_id, identity_name, sample_context, distance = rows[0]

    # Find the closest sample that belongs to a *different* identity (runner-up).
    second_distance: float | None = None
    if len(rows) > 1:
        for row in rows[1:]:
            if row[1] != identity_id:
                second_distance = float(row[4])
                break
        # If both rows share the same identity, the second_distance stays None.
        # In that case there is no competing identity, so no margin check needed.

    # Check whether the winning match is sufficiently separated from the
    # runner-up identity.  If not, downgrade the confidence level.
    margin_ok = (second_distance is None) or (second_distance - float(distance) >= min_margin)

    if float(distance) <= match_threshold:
        if margin_ok:
            status = "matched"
        elif float(distance) <= suggest_threshold:
            status = "suggest"   # too close to a rival – downgrade
        else:
            status = "unknown"
    elif float(distance) <= suggest_threshold:
        status = "suggest"
    else:
        status = "unknown"

    if status == "unknown":
        return {"status": "unknown", "identity_id": None, "identity_name": None,
                "sample_id": None, "sample_context": None,
                "distance": float(distance), "second_distance": second_distance}

    return {
        "status": status,
        "identity_id": identity_id,
        "identity_name": identity_name,
        "sample_id": sample_id,
        "sample_context": sample_context,
        "distance": float(distance),
        "second_distance": second_distance,
    }


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
        "is_low_quality",
    )
    data = {k: v for k, v in kwargs.items() if k in _ALLOWED_COLS}
    if "match_distance" in data:
        data["match_distance"] = _sanitize_float(data["match_distance"])
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


def get_existing_segment(
    conn: mariadb.Connection,
    series_name: str,
    episode_title: str,
    start_ms: int,
    end_ms: int,
) -> dict | None:
    """Return an existing segment row that matches the given timecodes, or None."""
    cur = conn.cursor()
    cur.execute(
        """SELECT id, identity_id, is_suggestion, match_distance
           FROM episode_segments
           WHERE series_name = ? AND episode_title = ?
             AND start_ms = ? AND end_ms = ?
           LIMIT 1""",
        (series_name, episode_title, start_ms, end_ms),
    )
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def update_segment_match(
    conn: mariadb.Connection,
    segment_id: int,
    identity_id: int | None,
    matched_sample_id: int | None,
    match_distance: float | None,
    is_suggestion: bool,
    embedding: bytes | None = None,
    transcript: str | None = None,
    is_low_quality: bool = False,
) -> None:
    """Update only the auto-detected fields; preserve manual identity assignments."""
    match_distance = _sanitize_float(match_distance)
    cur = conn.cursor()
    if embedding is not None and transcript is not None:
        cur.execute(
            """UPDATE episode_segments
               SET identity_id = ?, matched_sample_id = ?, match_distance = ?,
                   is_suggestion = ?, embedding = ?, transcript = ?, is_low_quality = ?
               WHERE id = ?""",
            (identity_id, matched_sample_id, match_distance, is_suggestion,
             embedding, transcript, is_low_quality, segment_id),
        )
    else:
        cur.execute(
            """UPDATE episode_segments
               SET identity_id = ?, matched_sample_id = ?, match_distance = ?,
                   is_suggestion = ?, is_low_quality = ?
               WHERE id = ?""",
            (identity_id, matched_sample_id, match_distance, is_suggestion,
             is_low_quality, segment_id),
        )
    conn.commit()


def update_segment_identity(
    conn: mariadb.Connection,
    segment_id: int,
    identity_id: int,
    matched_sample_id: int | None = None,
    match_distance: float | None = None,
    is_suggestion: bool = False,
) -> None:
    match_distance = _sanitize_float(match_distance)
    cur = conn.cursor()
    cur.execute(
        """UPDATE episode_segments
           SET identity_id = ?, matched_sample_id = ?,
               match_distance = ?, is_suggestion = ?
           WHERE id = ?""",
        (identity_id, matched_sample_id, match_distance, is_suggestion, segment_id),
    )
    conn.commit()


def update_segment_tts_path(conn: mariadb.Connection, segment_id: int, wav_path: str) -> None:
    """Store the path to the extracted TTS WAV snippet for a segment."""
    cur = conn.cursor()
    cur.execute(
        "UPDATE episode_segments SET tts_wav_path = ? WHERE id = ?",
        (wav_path, segment_id),
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


def get_segment(conn: mariadb.Connection, segment_id: int) -> dict | None:
    """Return full segment row as dict, or None if not found."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT es.id, es.series_name, es.episode_title, es.video_path,
               es.start_ms, es.end_ms, es.speaker_label,
               es.identity_id, i.name AS identity_name,
               es.transcript, es.is_low_quality, es.tts_wav_path
        FROM episode_segments es
        LEFT JOIN identities i ON i.id = es.identity_id
        WHERE es.id = ?
        """,
        (segment_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


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
            es.match_distance, es.transcript, es.confidence, es.is_suggestion,
            es.is_low_quality, es.tts_wav_path
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
