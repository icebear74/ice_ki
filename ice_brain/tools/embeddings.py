"""
Embedding helper – loads paraphrase-multilingual-mpnet-base-v2 (768-dim)
via sentence-transformers and provides encode / similarity / pack helpers for
storing embeddings in the MariaDB VECTOR(768) columns defined in schema.sql.

Singleton pattern: the model is loaded once (either eagerly via
load_embedding_model() at server startup or on first call to embed()) and
kept in memory for the lifetime of the process.

Public API
----------
load_embedding_model()  -> None        eager startup load; logs error prominently on failure
embed(texts)            -> np.ndarray  shape (N, 768), float32, L2-normalised
embed_one(text)         -> np.ndarray  shape (768,),   float32, L2-normalised
cosine_similarity       -> float       dot product of two normalised vectors
pack_embedding          -> bytes       LE float32 bytes for DB (3072 B)
vec_to_text             -> str         JSON-array string for VEC_FromText() INSERT/UPDATE
unpack_embedding        -> np.ndarray  bytes → float32 array
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

# 768-dim multilingual model – matches VECTOR(768) columns in schema.sql.
# First use triggers a ~1 GB download from HuggingFace (cached afterwards).
_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM = 768

_lock = threading.Lock()
_model = None   # SentenceTransformer instance, populated on first use
_device = "cpu"  # Target device; override via configure_embedding_device() at startup


def configure_embedding_device(device: str) -> None:
    """Set the PyTorch device used when loading the embedding model.

    Must be called *before* the model is first loaded (i.e. at server startup,
    before the first call to embed() / load_embedding_model()).  Typical values:
      - "cpu"      – default, no GPU
      - "cuda"     – first available CUDA GPU
      - "cuda:0"   – specific CUDA device index 0
      - "cuda:1"   – specific CUDA device index 1  (e.g. the P4)
    Has no effect if the model is already loaded.
    """
    global _device  # noqa: PLW0603
    _device = device
    logger.debug("Embedding model device configured: %s", device)


def _get_model():
    """Return the SentenceTransformer singleton, loading it on first call."""
    global _model  # noqa: PLW0603
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:  # double-checked inside lock
            return _model
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        logger.info("Loading embedding model '%s' on device '%s' …", _MODEL_NAME, _device)
        _model = SentenceTransformer(_MODEL_NAME, device=_device)
        logger.info(
            "Embedding model '%s' loaded (%d-dim, device=%s).",
            _MODEL_NAME, EMBEDDING_DIM, _device,
        )
    return _model


def _fallback_to_cpu() -> None:
    """Move the loaded embedding model to CPU after a CUDA OOM.

    Called automatically by :func:`embed` when the GPU runs out of memory.
    After this all subsequent calls will run on CPU for the remainder of the
    process lifetime.
    """
    global _device  # noqa: PLW0603
    logger.warning(
        "Embedding model: CUDA out of memory – moving model to CPU "
        "(all subsequent embedding calls will use CPU)."
    )
    _device = "cpu"
    if _model is not None:
        try:
            _model.to("cpu")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding model: could not move to CPU: %s", exc)
    try:
        import torch  # noqa: PLC0415
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def load_embedding_model() -> bool:
    """Eagerly load the embedding model singleton.

    Intended to be called once at server startup so that download/disk errors
    are visible immediately rather than on the first wiki search request.

    Returns True on success, False on failure (error is logged prominently).
    The server continues to run even if this fails; wiki search is simply
    unavailable until the model can be loaded.
    """
    try:
        _get_model()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("=" * 60)
        logger.error("EMBEDDING MODEL LOAD FAILED: %s", exc)
        logger.error(
            "Wiki-Vektorsuche ist nicht verfügbar bis das Modell geladen werden kann."
        )
        logger.error(
            "Modell: %s (~1.1 GB) – ausreichend freien Speicher sicherstellen.", _MODEL_NAME
        )
        logger.error("=" * 60)
        return False


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def embed(texts: list[str]) -> np.ndarray:
    """Encode *texts* into L2-normalised float32 embeddings.

    Returns shape (N, 768) numpy array.

    If the configured GPU runs out of memory the model is transparently moved
    to CPU and the encoding is retried.  All subsequent calls will also use CPU.
    """
    model = _get_model()
    try:
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and _device != "cpu":
            _fallback_to_cpu()
            model = _get_model()
            vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        else:
            raise
    return np.array(vecs, dtype=np.float32)


def embed_one(text: str) -> np.ndarray:
    """Encode a single string into a normalised (768,) float32 embedding."""
    return embed([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two L2-normalised vectors (= their dot product)."""
    return float(np.dot(a, b))


def pack_embedding(vec: np.ndarray) -> bytes:
    """Pack a float32 numpy array into LE bytes for MariaDB VECTOR storage."""
    return np.array(vec, dtype=np.float32).tobytes()


def vec_to_text(vec: np.ndarray) -> str:
    """Serialise a float32 vector as a JSON-array string for ``VEC_FromText()``.

    MariaDB's VECTOR type accepts the ``[f1,f2,...]`` text format via
    ``VEC_FromText()``.  Passing raw bytes through ``mysql-connector-python``
    can cause charset-encoding corruption (the connector applies utf8mb4
    encoding to ``bytes`` parameters), so text format is the safe path for
    INSERT/UPDATE statements.
    """
    return "[" + ",".join(f"{v:.8g}" for v in np.array(vec, dtype=np.float32).tolist()) + "]"


def unpack_embedding(data: bytes) -> np.ndarray:
    """Unpack LE bytes retrieved from MariaDB VECTOR / MEDIUMBLOB column."""
    return np.frombuffer(bytes(data), dtype=np.float32).copy()
