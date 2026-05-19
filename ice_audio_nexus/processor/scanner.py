"""
Step-1 visual person discovery scanner.

Pipeline:
1. Sample frames (default 4 FPS)
2. Detect faces
3. Build local tracks with IoU + descriptor similarity
4. Promote only clear tracks (high precision)
5. Persist detections, tracks, representative crops and overlay data
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from db.database import (  # noqa: E402
    assign_detection_to_track,
    clear_video_scan_data,
    create_face_detection,
    create_face_track,
    create_visual_group,
    create_visual_seed,
    ensure_schema,
    get_connection,
    list_videos,
    list_visual_groups,
    rebuild_overlay_for_video,
    run_expansion_for_group,
    set_video_scan_status,
    upsert_production_and_video,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom < 1e-9:
        return -1.0
    return float(np.dot(vec_a, vec_b) / denom)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def _probe_duration_ms(path: str) -> int | None:
    import json
    import subprocess

    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        data = json.loads(out.stdout or "{}")
    except Exception:
        return None
    duration = float(data.get("format", {}).get("duration", 0) or 0)
    for stream in data.get("streams", []):
        try:
            duration = max(duration, float(stream.get("duration", 0) or 0))
        except (TypeError, ValueError):
            pass
    if duration <= 0:
        return None
    return int(duration * 1000.0)


def _parse_episode_code(title_or_path: str) -> str | None:
    m = re.search(r"(S\d{1,2}E\d{1,2})", title_or_path, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _parse_season_label(title_or_path: str) -> str | None:
    """Extract a season label like 'S01' from a filename containing SxxEyy."""
    m = re.search(r"(S\d{1,2})E\d{1,2}", title_or_path, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _parse_production_from_path(path: Path) -> str:
    if path.parent.name:
        return path.parent.name
    return "Unknown Production"


#: Video file extensions recognised when scanning a directory.
_SCAN_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4", ".mp2", ".mkv", ".avi", ".mov", ".m4v",
        ".webm", ".ts", ".mpg", ".mpeg", ".wmv", ".flv",
        ".vob", ".m2v", ".m2ts", ".mts", ".divx", ".3gp",
    }
)

def _resolve_input_video_path(video_path: str) -> Path:
    """
    Resolve input path and recover from common escaped-space CLI misuse.
    """
    source = Path(video_path).expanduser().resolve()
    if source.exists():
        return source

    if "\\ " in video_path:
        normalized = video_path.replace("\\ ", " ")
        candidate = Path(normalized).expanduser().resolve()
        if candidate.exists():
            logger.warning(
                "Input path contained literal '\\ ' escapes; normalized to: %s",
                candidate,
            )
            return candidate

    raise FileNotFoundError(
        "Video not found: "
        f"{source}. If your path contains spaces, either quote it without backslashes "
        "(\"/path/with spaces/file.mkv\") or use backslashes without quotes."
    )


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _configure_torch_model_cache(models_dir: Path) -> dict[str, str]:
    models_dir.mkdir(parents=True, exist_ok=True)
    torch_home = models_dir / "torch_home"
    hf_home = models_dir / "huggingface"
    torch_home.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_home))
    os.environ.setdefault("HF_HOME", str(hf_home))
    return {"torch_home": str(torch_home), "hf_home": str(hf_home)}


def _cuda_device_supported(device_idx: int) -> tuple[bool, str]:
    """Return (supported, reason) for *device_idx* against the current torch wheel.

    PyTorch silently accepts the device selection even when a GPU's compute
    capability (CC) is below what the wheel was compiled for.  The failure only
    surfaces at the first CUDA kernel launch ("no kernel image available").
    This probe compares the device CC against the set of CCs the wheel was
    actually built for so we can fall back to CPU before that happens.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "cuda_unavailable"

        cc = torch.cuda.get_device_capability(device_idx)  # (major, minor)
        device_cc = cc[0] * 10 + cc[1]  # e.g. 6.0 → 60

        # The set of compiled-in CCs is exposed via _C._cuda_getCompiledCudaArches()
        # on recent torch, but that API is not always present.  Fall back to the
        # documented per-version check.
        supported_ccs: set[int] = set()
        try:
            # Available since torch 2.0
            arches = torch.cuda._get_device_properties  # type: ignore[attr-defined]
            _ = arches  # just a probe; use the compiled-arches API below
            compiled: list[int] = [int(a) for a in torch.cuda.get_arch_list()]  # "sm_86" → 86
            # Each compiled SM supports all devices with that SM or above up to
            # the next listed SM.  Build a supported-CC set: a device at cc X is
            # supported when X >= any compiled SM in the wheel (torch selects the
            # closest compatible binary / PTX fallback).
            # Practically: sm_60 requires the wheel to ship a sm_60 or sm_61
            # cubin or a PTX that can JIT to it.  Wheels cu124+ omit sm_60/sm_61.
            supported_ccs = {int(a) for a in compiled}
        except Exception:  # noqa: BLE001
            pass

        if supported_ccs:
            # A device is usable if the wheel contains a cubin/PTX at or below
            # the device's CC (torch can JIT-compile PTX upward, but requires
            # the PTX to have been generated for a CC ≤ device_cc).
            compatible = any(wcc <= device_cc for wcc in supported_ccs)
            if not compatible:
                cc_str = f"{cc[0]}.{cc[1]}"
                wcc_str = ", ".join(f"sm_{c}" for c in sorted(supported_ccs))
                reason = (
                    f"GPU CC {cc_str} (sm_{device_cc}) is not supported by this "
                    f"torch wheel (supports: {wcc_str}). "
                    "Re-run setup_env.sh to install a compatible wheel "
                    "(e.g. torch==2.4.1+cu121 for Pascal/Maxwell GPUs)."
                )
                return False, reason
        # If we couldn't determine supported CCs, optimistically allow it
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return True, f"cc_probe_failed ({exc})"


def _resolve_cuda_index(requested_device: str | None, gpu_device_id: int) -> tuple[int | None, str]:
    raw = (requested_device or "").strip().lower()
    if not raw:
        return gpu_device_id, f"cuda:{gpu_device_id}"
    if raw == "cpu":
        return None, "cpu"
    if raw == "cuda":
        return 0, "cuda:0"
    m = re.fullmatch(r"cuda:(\d+)", raw)
    if m:
        return int(m.group(1)), raw
    return gpu_device_id, f"invalid:{requested_device}"


def _torch_runtime_for_component(
    *,
    prefer_gpu: bool,
    requested_device: str | None,
    gpu_device_id: int,
    component_name: str,
) -> tuple["torch.device", dict[str, object]]:
    import torch

    requested_idx, requested_label = _resolve_cuda_index(requested_device, gpu_device_id)
    diagnostics: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count() if torch.cuda.is_available() else 0),
        "component": component_name,
        "requested_device": requested_label,
        "requested_device_id": gpu_device_id,
        "prefer_gpu": prefer_gpu,
    }
    visible_devices: list[dict[str, object]] = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            name = "unknown"
            cc_str = "unknown"
            try:
                name = str(torch.cuda.get_device_name(idx))
                cc = torch.cuda.get_device_capability(idx)
                cc_str = f"{cc[0]}.{cc[1]}"
            except Exception:  # noqa: BLE001
                pass
            visible_devices.append({"index": idx, "name": name, "compute_capability": cc_str})
    diagnostics["cuda_devices"] = visible_devices

    if not prefer_gpu:
        diagnostics["selected_device"] = "cpu"
        diagnostics["selected_accelerator"] = "cpu"
        diagnostics["selection_reason"] = "gpu_disabled"
        return torch.device("cpu"), diagnostics

    if requested_idx is None:
        diagnostics["selected_device"] = "cpu"
        diagnostics["selected_accelerator"] = "cpu"
        diagnostics["selection_reason"] = "cpu_requested"
        return torch.device("cpu"), diagnostics

    if str(requested_label).startswith("invalid:"):
        diagnostics["selected_device"] = "cpu"
        diagnostics["selected_accelerator"] = "cpu"
        diagnostics["selection_reason"] = "invalid_device_string"
        return torch.device("cpu"), diagnostics

    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        diagnostics["selected_device"] = "cpu"
        diagnostics["selected_accelerator"] = "cpu"
        diagnostics["selection_reason"] = "cuda_unavailable"
        return torch.device("cpu"), diagnostics

    selected_idx = requested_idx if 0 <= requested_idx < torch.cuda.device_count() else 0
    if selected_idx != requested_idx:
        diagnostics["selection_reason"] = "gpu_device_id_out_of_range"
    else:
        diagnostics["selection_reason"] = "cuda_selected"

    # Check compute-capability compatibility before committing to CUDA
    supported, cc_reason = _cuda_device_supported(selected_idx)
    if not supported:
        logger.warning(
            "Falling back to CPU: %s",
            cc_reason,
        )
        diagnostics["selected_device"] = "cpu"
        diagnostics["selected_accelerator"] = "cpu"
        diagnostics["selection_reason"] = "cc_incompatible"
        diagnostics["cc_incompatible_reason"] = cc_reason
        return torch.device("cpu"), diagnostics

    try:
        torch.cuda.set_device(selected_idx)
    except Exception as exc:  # noqa: BLE001
        diagnostics["selected_device"] = "cpu"
        diagnostics["selected_accelerator"] = "cpu"
        diagnostics["selection_reason"] = "cuda_set_device_failed"
        diagnostics["cuda_set_device_error"] = str(exc)
        return torch.device("cpu"), diagnostics

    diagnostics["selected_device"] = f"cuda:{selected_idx}"
    diagnostics["selected_accelerator"] = "gpu"
    return torch.device(f"cuda:{selected_idx}"), diagnostics


def _torch_runtime(prefer_gpu: bool, gpu_device_id: int) -> tuple["torch.device", dict[str, object]]:
    return _torch_runtime_for_component(
        prefer_gpu=prefer_gpu,
        requested_device=None,
        gpu_device_id=gpu_device_id,
        component_name="shared",
    )


def _torch_diagnostics(prefer_gpu: bool, gpu_device_id: int) -> dict[str, object]:
    import cv2

    _det_device, detector_diag = _torch_runtime_for_component(
        prefer_gpu=prefer_gpu,
        requested_device=os.getenv("FACE_DETECTOR_DEVICE"),
        gpu_device_id=gpu_device_id,
        component_name="detector",
    )
    _ver_device, verifier_diag = _torch_runtime_for_component(
        prefer_gpu=prefer_gpu,
        requested_device=os.getenv("FACE_VERIFIER_DEVICE"),
        gpu_device_id=gpu_device_id,
        component_name="verifier",
    )
    _emb_device, embedding_diag = _torch_runtime_for_component(
        prefer_gpu=prefer_gpu,
        requested_device=os.getenv("FACE_EMBEDDING_DEVICE"),
        gpu_device_id=gpu_device_id,
        component_name="embedding",
    )
    diagnostics = {
        "opencv_version": cv2.__version__,
        "detector": detector_diag,
        "verifier": verifier_diag,
        "embedding": embedding_diag,
    }
    data_root = Path(os.getenv("FACE_DATA_DIR", "data/faces")).resolve()
    models_dir = Path(os.getenv("FACE_MODELS_DIR", str(data_root / "models"))).resolve()
    diagnostics["model_cache"] = _configure_torch_model_cache(models_dir)
    return diagnostics


class _TorchMTCNNDetector:
    def __init__(self, mtcnn_model):
        self._model = mtcnn_model

    def detect(
        self,
        frame: np.ndarray,
        min_confidence: float,
        min_size_px: int,
    ) -> list[tuple[int, int, int, int, float]]:
        import cv2

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = self._model.detect(rgb)
        if boxes is None or probs is None:
            return []

        results: list[tuple[int, int, int, int, float]] = []
        for box, score in zip(boxes, probs):
            if box is None:
                continue
            confidence = float(score or 0.0)
            if confidence < min_confidence:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            fw, fh = x2 - x1, y2 - y1
            if fw < min_size_px or fh < min_size_px:
                continue
            results.append((x1, y1, fw, fh, confidence))
        return results


class _TorchMTCNNVerifier:
    def __init__(self, mtcnn_model):
        self._model = mtcnn_model

    def verify(
        self,
        crop: np.ndarray,
        *,
        score_threshold: float,
        min_area_ratio: float,
        max_center_offset: float,
    ) -> tuple[bool, dict[str, object]]:
        import cv2

        h, w = crop.shape[:2]
        if h < 20 or w < 20:
            return False, {"score": 0.0, "area_ratio": 0.0, "center_offset": 1.0, "reason": "too_small"}

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        boxes, probs = self._model.detect(rgb)
        if boxes is None or probs is None or len(boxes) == 0:
            return False, {"score": 0.0, "area_ratio": 0.0, "center_offset": 1.0, "reason": "no_face"}

        best_idx = int(np.argmax(np.asarray(probs, dtype=np.float32)))
        best_box = boxes[best_idx]
        score = float(probs[best_idx] or 0.0)

        vx1, vy1, vx2, vy2 = [float(v) for v in best_box]
        vw, vh = max(0.0, vx2 - vx1), max(0.0, vy2 - vy1)
        area_ratio = max(0.0, (vw * vh) / float(max(1, w * h)))
        cx = vx1 + (vw / 2.0)
        cy = vy1 + (vh / 2.0)
        center_offset = (abs(cx - (w / 2.0)) / max(1.0, w)) + (abs(cy - (h / 2.0)) / max(1.0, h))

        reason = "ok"
        if score < score_threshold:
            reason = "low_score"
        elif area_ratio < min_area_ratio:
            reason = "low_area_ratio"
        elif center_offset > max_center_offset:
            reason = "high_center_offset"

        passed = reason == "ok"
        return passed, {"score": score, "area_ratio": area_ratio, "center_offset": center_offset, "reason": reason}


def _create_mtcnn_model(
    *,
    device: "torch.device",
    min_face_size: int,
    thresholds: tuple[float, float, float],
):
    try:
        from facenet_pytorch import MTCNN
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'facenet-pytorch'. Run ice_audio_nexus/setup_env.sh to install Torch models."
        ) from exc

    return MTCNN(
        image_size=160,
        margin=0,
        min_face_size=max(20, int(min_face_size)),
        thresholds=list(thresholds),
        factor=0.709,
        post_process=False,
        keep_all=True,
        device=device,
    )


def _get_torch_face_detector(
    models_dir: Path,
    *,
    device: "torch.device",
    min_face_size_px: int,
):
    _configure_torch_model_cache(models_dir)
    try:
        model = _create_mtcnn_model(
            device=device,
            min_face_size=min_face_size_px,
            thresholds=(0.6, 0.7, 0.7),
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize Torch face detector model. "
            f"Check internet connectivity for first-time model download or pre-populate {models_dir / 'torch_home'}: {exc}"
        ) from exc
    backend = "gpu" if str(device).startswith("cuda") else "cpu"
    logger.info("Face detector: Torch MTCNN active (%s / %s)", backend, device)
    return _TorchMTCNNDetector(model), {"backend": "torch", "target": str(device), "accelerator": backend}


def _get_torch_face_verifier_model(
    models_dir: Path,
    *,
    enabled: bool,
    device: "torch.device",
):
    if not enabled:
        logger.info("Face verifier: disabled")
        return None, {"enabled": False, "backend": "disabled"}

    _configure_torch_model_cache(models_dir)
    try:
        model = _create_mtcnn_model(
            device=device,
            min_face_size=20,
            thresholds=(0.7, 0.8, 0.8),
        )
    except Exception as exc:
        logger.warning(
            "Face verifier unavailable: Torch model init/download failed. "
            "Pre-populate cache in %s or check internet access. Error: %s",
            models_dir / "torch_home",
            exc,
        )
        return None, {"enabled": False, "backend": "init_failed", "error": str(exc)}

    backend = "gpu" if str(device).startswith("cuda") else "cpu"
    logger.info("Face verifier: enabled (torch / %s)", device)
    return _TorchMTCNNVerifier(model), {"enabled": True, "backend": "torch", "target": str(device), "accelerator": backend}


def _get_facenet_embedding_model(
    models_dir: Path,
    *,
    device: "torch.device",
):
    """Load InceptionResnetV1 (VGGFace2) for 512-dim L2-normalised face embeddings.

    Returns (model, info_dict).  On any failure returns (None, {'enabled': False, ...})
    so the scanner can fall back to the simple pixel-hash descriptor.
    """
    _configure_torch_model_cache(models_dir)
    try:
        from facenet_pytorch import InceptionResnetV1
    except ImportError:
        logger.warning("InceptionResnetV1 not available – falling back to pixel-hash descriptor")
        return None, {"enabled": False, "reason": "facenet_pytorch missing InceptionResnetV1"}

    try:
        model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load InceptionResnetV1 (will use fallback descriptor): %s", exc
        )
        return None, {"enabled": False, "reason": str(exc)}

    backend = "gpu" if str(device).startswith("cuda") else "cpu"
    logger.info("FaceNet embedding model: InceptionResnetV1 (vggface2) active (%s / %s)", backend, device)
    return model, {"enabled": True, "backend": "facenet_vggface2", "target": str(device), "accelerator": backend}


def _build_face_embedding(
    crop_bgr: np.ndarray,
    facenet_model,
    device: "torch.device",
) -> np.ndarray:
    """Return a 512-dim L2-normalised FaceNet embedding for *crop_bgr*.

    Falls back to the simple 128-dim pixel-hash descriptor when *facenet_model*
    is None (model load failed) so the pipeline never stalls.
    """
    import cv2

    if facenet_model is None:
        return _build_face_descriptor_fallback(crop_bgr)

    try:
        import torch

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        face = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_CUBIC)
        tensor = torch.from_numpy(face.astype(np.float32)).permute(2, 0, 1)
        tensor = (tensor - 127.5) / 128.0
        tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = facenet_model(tensor)
        return embedding.squeeze(0).cpu().numpy()
    except Exception as exc:  # noqa: BLE001
        logger.warning("FaceNet embedding failed (using fallback): %s", exc)
        return _build_face_descriptor_fallback(crop_bgr)


def _build_face_descriptor_fallback(crop_bgr: np.ndarray) -> np.ndarray:
    """128-dim grayscale pixel-hash – used only when FaceNet is unavailable."""
    import cv2

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 8), interpolation=cv2.INTER_AREA)
    desc = small.astype(np.float32).reshape(-1) / 255.0
    norm = np.linalg.norm(desc)
    if norm > 1e-9:
        desc = desc / norm
    return desc


def _torch_detect_faces(
    detector: _TorchMTCNNDetector,
    frame: np.ndarray,
    min_confidence: float,
    min_size_px: int,
) -> list[tuple[int, int, int, int, float]]:
    """Run the Torch face detector on *frame*.

    Returns a list of ``(x, y, w, h, confidence)`` tuples for every detected
    face that passes the confidence and minimum-size filters.
    """
    return detector.detect(frame, min_confidence=min_confidence, min_size_px=min_size_px)


def _verify_face_candidate(
    verifier: _TorchMTCNNVerifier,
    crop: np.ndarray,
    *,
    score_threshold: float,
    min_area_ratio: float,
    max_center_offset: float,
) -> tuple[bool, dict[str, object]]:
    return verifier.verify(
        crop,
        score_threshold=score_threshold,
        min_area_ratio=min_area_ratio,
        max_center_offset=max_center_offset,
    )


@dataclass
class Detection:
    frame_index: int
    timestamp_ms: int
    bbox: tuple[int, int, int, int]
    confidence: float
    sharpness: float
    area_ratio: float
    embedding: np.ndarray
    crop_path: str
    db_id: int | None = None


@dataclass
class Track:
    detections: list[Detection] = field(default_factory=list)
    last_ts_ms: int = 0

    def append(self, detection: Detection) -> None:
        self.detections.append(detection)
        self.last_ts_ms = detection.timestamp_ms

    @property
    def first_ts_ms(self) -> int:
        return self.detections[0].timestamp_ms

    @property
    def last_bbox(self) -> tuple[int, int, int, int]:
        return self.detections[-1].bbox


def _load_existing_group_centroids(conn, production_id: int) -> list[dict]:
    groups = list_visual_groups(conn, production_id=production_id, include_seeds=True)
    out: list[dict] = []
    for g in groups:
        seeds = g.get("seeds") or []
        vectors: list[np.ndarray] = []
        for seed in seeds:
            emb = seed.get("embedding")
            if isinstance(emb, list) and emb:
                try:
                    vectors.append(np.asarray(emb, dtype=np.float32))
                except Exception:  # noqa: BLE001
                    continue
        if not vectors:
            continue
        centroid = np.mean(np.stack(vectors, axis=0), axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-9:
            centroid = centroid / norm
        out.append(
            {
                "id": int(g["id"]),
                "label": g["label"],
                "centroid": centroid,
                "count": len(vectors),
            }
        )
    return out


def scan_video(
    video_path: str,
    *,
    production: str | None = None,
    title: str | None = None,
    episode_code: str | None = None,
    sample_fps: float = 4.0,
    min_clear_seconds: float = 2.0,
    min_face_area_ratio: float = 0.04,
    min_sharpness: float = 70.0,
    min_stability: float = 0.30,
    track_max_gap_ms: int = 600,
    iou_threshold: float = 0.30,
    descriptor_threshold: float = 0.72,
    dnn_confidence: float = 0.65,
    min_face_size_px: int = 80,
    max_aspect_ratio_deviation: float = 0.65,
    min_brightness: float = 40.0,
    min_quality_score: float = 0.55,
    seed_acceptance_threshold: float = 0.60,
    verifier_enabled: bool = True,
    verifier_score_threshold: float = 0.92,
    verifier_min_area_ratio: float = 0.25,
    verifier_max_center_offset: float = 0.45,
    prefer_gpu: bool = True,
    gpu_device_id: int = 0,
    detector_device: str | None = None,
    verifier_device: str | None = None,
    embedding_device: str | None = None,
    gpu_diagnostics: bool = True,
    seed_group_similarity_threshold: float = 0.90,
    duplicate_similarity_threshold: float = 0.985,
    start_offset_seconds: float = 0.0,
    max_sampled_frames: int = 0,
    write_debug_stats: bool = False,
    debug_stats_dir: str | None = None,
) -> dict:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV not installed. Install opencv-python-headless.") from exc

    source = _resolve_input_video_path(video_path)

    production_name = production or _parse_production_from_path(source)
    video_title = title or source.stem
    ep_code = episode_code or _parse_episode_code(source.name)
    season_lbl = _parse_season_label(source.name)
    duration_ms = _probe_duration_ms(str(source))

    data_root = Path(os.getenv("FACE_DATA_DIR", "data/faces")).resolve()
    crops_root = data_root / "crops" / source.stem
    reps_root = data_root / "tracks" / source.stem

    ensure_schema()
    conn = get_connection()
    production_id, video_id = upsert_production_and_video(
        conn,
        production_title=production_name,
        video_title=video_title,
        video_path=str(source),
        season_label=season_lbl,
        episode_code=ep_code,
        duration_ms=duration_ms,
        production_meta={"scanner": "visual-step1"},
        video_meta={"sample_fps": sample_fps, "workflow": {"seed_scanned": True}},
    )

    logger.info("Scanning video_id=%s (%s)", video_id, source)
    set_video_scan_status(conn, video_id, "scanning")
    # clear_video_scan_data removes all stale DB rows AND deletes crops/tracks
    # image directories for this video stem – so create the dirs AFTER the call.
    clear_video_scan_data(conn, video_id)

    crops_root.mkdir(parents=True, exist_ok=True)
    reps_root.mkdir(parents=True, exist_ok=True)
    logger.info("Output directories ready: %s | %s", crops_root, reps_root)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        set_video_scan_status(conn, video_id, "failed")
        conn.close()
        raise RuntimeError(f"Could not open video: {source}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0:
        native_fps = 25.0
    frame_step = max(1, int(round(native_fps / max(sample_fps, 0.5))))
    start_offset_seconds = max(0.0, float(start_offset_seconds))
    max_sampled_frames = max(0, int(max_sampled_frames))

    # Load Torch face detector/verifier (downloads model weights once on first run).
    models_dir = Path(os.getenv("FACE_MODELS_DIR", str(data_root / "models"))).resolve()
    detector_torch_device, detector_diag = _torch_runtime_for_component(
        prefer_gpu=prefer_gpu,
        requested_device=detector_device,
        gpu_device_id=gpu_device_id,
        component_name="detector",
    )
    verifier_torch_device, verifier_diag = _torch_runtime_for_component(
        prefer_gpu=prefer_gpu,
        requested_device=verifier_device,
        gpu_device_id=gpu_device_id,
        component_name="verifier",
    )
    embedding_torch_device, embedding_diag = _torch_runtime_for_component(
        prefer_gpu=prefer_gpu,
        requested_device=embedding_device,
        gpu_device_id=gpu_device_id,
        component_name="embedding",
    )
    diag_info = {
        "opencv_version": cv2.__version__,
        "detector": detector_diag,
        "verifier": verifier_diag,
        "embedding": embedding_diag,
    }
    cache_info = _configure_torch_model_cache(models_dir)
    diag_info["model_cache"] = cache_info
    if gpu_diagnostics:
        logger.info("OpenCV version (I/O only): %s", cv2.__version__)
        logger.info(
            "Torch devices | detector=%s (%s) | verifier=%s (%s) | embedding=%s (%s)",
            detector_diag.get("selected_device"),
            detector_diag.get("selection_reason"),
            verifier_diag.get("selected_device"),
            verifier_diag.get("selection_reason"),
            embedding_diag.get("selected_device"),
            embedding_diag.get("selection_reason"),
        )
        logger.info("Model cache: TORCH_HOME=%s | HF_HOME=%s", cache_info.get("torch_home"), cache_info.get("hf_home"))

    try:
        face_detector, detector_runtime = _get_torch_face_detector(
            models_dir,
            device=detector_torch_device,
            min_face_size_px=min_face_size_px,
        )
    except Exception as exc:
        set_video_scan_status(conn, video_id, "failed")
        conn.close()
        cap.release()
        raise RuntimeError(f"Could not load Torch face detector: {exc}") from exc

    verifier, verifier_runtime = _get_torch_face_verifier_model(
        models_dir,
        enabled=verifier_enabled,
        device=verifier_torch_device,
    )

    # FaceNet embedding model (InceptionResnetV1) – generates 512-dim embeddings for
    # proper face re-identification. Falls back to 128-dim pixel-hash on load failure.
    facenet_model, facenet_runtime = _get_facenet_embedding_model(models_dir, device=embedding_torch_device)
    if gpu_diagnostics:
        logger.info(
            "FaceNet embedding: enabled=%s | backend=%s",
            facenet_runtime.get("enabled"),
            facenet_runtime.get("backend", "pixel_hash_fallback"),
        )

    start_frame_index = 0
    if start_offset_seconds > 0.0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_offset_seconds * 1000.0)
        start_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or int(start_offset_seconds * native_fps))
    frame_index = max(-1, start_frame_index - 1)
    timestamp_ms = int(start_offset_seconds * 1000.0)
    sampled_frames = 0
    detections_considered = 0
    seeds_accepted = 0
    groups_created = 0
    groups_matched = 0
    pseudo_tracks = 0
    quality_passed_before_verifier = 0
    verifier_rejected_after_quality = 0
    duplicate_matches = 0
    reject_reasons: dict[str, int] = {
        "small": 0,
        "blurry": 0,
        "pose": 0,
        "occluded": 0,
        "dark": 0,
        "quality_score": 0,
        "verifier": 0,
        "duplicate": 0,
        "other": 0,
    }
    group_centroids = _load_existing_group_centroids(conn, production_id)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_sample_frames = max(1, total_frames // frame_step) if total_frames > 0 else None
    duration_s_est = (total_frames / native_fps) if (native_fps > 0 and total_frames > 0) else None
    if total_sample_frames and start_frame_index > 0:
        total_sample_frames = max(1, total_sample_frames - (start_frame_index // frame_step))
    if max_sampled_frames > 0:
        total_sample_frames = min(total_sample_frames, max_sampled_frames) if total_sample_frames else max_sampled_frames
    _last_pct_reported = -1  # tracks last reported percentage milestone

    if duration_s_est:
        logger.info(
            "Starting seed discovery: %s | duration ~%.0fs | ~%d frames to sample at %.1f fps | start_offset=%.1fs",
            source.name, duration_s_est, total_sample_frames or 0, sample_fps, start_offset_seconds,
        )
    else:
        logger.info("Starting seed discovery: %s | sampling at %.1f fps | start_offset=%.1fs", source.name, sample_fps, start_offset_seconds)

    while True:
        if max_sampled_frames > 0 and sampled_frames >= max_sampled_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % frame_step != 0:
            continue

        sampled_frames += 1
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            continue
        frame_area = float(w * h)
        timestamp_ms = int((frame_index / native_fps) * 1000.0)

        raw_faces = _torch_detect_faces(face_detector, frame, dnn_confidence, min_face_size_px)
        for i, (x, y, fw, fh, conf) in enumerate(raw_faces):
            detections_considered += 1
            fw = int(min(fw, w - x))
            fh = int(min(fh, h - y))
            if fw <= 0 or fh <= 0:
                reject_reasons["other"] += 1
                continue
            # Skip detections with extreme aspect ratios – real faces are roughly square.
            if fw < min_face_size_px or fh < min_face_size_px:
                reject_reasons["small"] += 1
                continue
            aspect = fw / max(fh, 1)
            if abs(aspect - 1.0) > max_aspect_ratio_deviation:
                reject_reasons["pose"] += 1
                continue
            crop = frame[y : y + fh, x : x + fw]
            if crop.size == 0:
                reject_reasons["other"] += 1
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            area_ratio = (fw * fh) / frame_area
            brightness = float(np.mean(gray))
            if area_ratio < min_face_area_ratio:
                reject_reasons["small"] += 1
                continue
            if sharpness < min_sharpness:
                reject_reasons["blurry"] += 1
                continue
            if brightness < min_brightness:
                reject_reasons["dark"] += 1
                continue

            normalized_sharpness = min(sharpness / max(min_sharpness * 2.0, 1.0), 1.0)
            normalized_area = min(area_ratio / max(min_face_area_ratio * 2.0, 1e-6), 1.0)
            normalized_brightness = min(brightness / 255.0, 1.0)
            quality_score = (
                float(conf) * 0.35
                + normalized_sharpness * 0.30
                + normalized_area * 0.20
                + normalized_brightness * 0.15
            )
            if quality_score < min_quality_score or quality_score < seed_acceptance_threshold:
                reject_reasons["quality_score"] += 1
                continue

            quality_passed_before_verifier += 1
            verifier_meta: dict[str, object] = {"score": 0.0, "area_ratio": 0.0, "center_offset": 0.0, "reason": "disabled"}
            if verifier is not None:
                try:
                    verified, verifier_meta = _verify_face_candidate(
                        verifier,
                        crop,
                        score_threshold=verifier_score_threshold,
                        min_area_ratio=verifier_min_area_ratio,
                        max_center_offset=verifier_max_center_offset,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Face verifier failed on frame=%d det=%d: %s", frame_index, i, exc)
                    verified = False
                    verifier_meta = {"score": 0.0, "area_ratio": 0.0, "center_offset": 1.0, "reason": "verifier_error"}
                if not verified:
                    verifier_rejected_after_quality += 1
                    verifier_reason = str(verifier_meta.get("reason") or "verifier_reject")
                    if verifier_reason in {"high_center_offset"}:
                        reject_reasons["pose"] += 1
                    elif verifier_reason in {"no_face", "low_area_ratio", "too_small"}:
                        reject_reasons["occluded"] += 1
                    else:
                        reject_reasons["verifier"] += 1
                    continue

            descriptor = _build_face_embedding(crop, facenet_model, embedding_torch_device)
            if descriptor.size == 0:
                reject_reasons["other"] += 1
                continue

            crop_name = f"f{frame_index:08d}_d{i:02d}.jpg"
            crop_path = crops_root / crop_name
            cv2.imwrite(str(crop_path), crop)

            det = Detection(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                bbox=(x, y, fw, fh),
                confidence=conf,
                sharpness=sharpness,
                area_ratio=area_ratio,
                embedding=descriptor,
                crop_path=_safe_relpath(crop_path, data_root),
            )
            det.db_id = create_face_detection(
                conn,
                video_id=video_id,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                bbox_x=x,
                bbox_y=y,
                bbox_w=fw,
                bbox_h=fh,
                confidence=det.confidence,
                sharpness=sharpness,
                crop_image_path=det.crop_path,
                embedding=descriptor.astype(float).tolist(),
                metadata={"area_ratio": area_ratio, "verifier": verifier_meta},
            )
            pseudo_tracks += 1
            track_id = create_face_track(
                conn,
                video_id=video_id,
                start_ms=timestamp_ms,
                end_ms=timestamp_ms,
                frame_count=1,
                mean_face_area=area_ratio,
                mean_sharpness=sharpness,
                mean_confidence=conf,
                stability_score=1.0,
                quality_score=quality_score,
                relevance_score=min(1.0, 0.4 + 0.6 * area_ratio / max(min_face_area_ratio, 1e-6)),
                is_clear=True,
                status="candidate",
                representative_image_path=det.crop_path,
                embedding=descriptor.astype(float).tolist(),
                metadata={
                    "duration_seconds": 0.0,
                    "seed_workflow": {
                        "mode": "seed_first",
                        "stage": "review",
                        "review_state": "pending",
                        "group_label": None,
                        "expansion_state": "blocked",
                        "notes": None,
                    },
                    "tracking_role": "seed_observation",
                    "thresholds": {
                        "min_face_area_ratio": min_face_area_ratio,
                        "min_sharpness": min_sharpness,
                        "seed_group_similarity_threshold": seed_group_similarity_threshold,
                        "verifier_enabled": verifier is not None,
                        "verifier_score_threshold": verifier_score_threshold,
                        "min_brightness": min_brightness,
                        "min_quality_score": min_quality_score,
                        "seed_acceptance_threshold": seed_acceptance_threshold,
                        "duplicate_similarity_threshold": duplicate_similarity_threshold,
                    },
                    "runtime": {
                        "detector_backend": detector_runtime.get("backend"),
                        "detector_target": detector_runtime.get("target"),
                        "verifier_backend": verifier_runtime.get("backend"),
                        "verifier_target": verifier_runtime.get("target"),
                        "embedding_backend": facenet_runtime.get("backend", "pixel_hash_fallback"),
                        "embedding_target": str(embedding_torch_device),
                    },
                },
            )
            assign_detection_to_track(conn, det.db_id, track_id)

            best_group = None
            best_sim = -1.0
            for g in group_centroids:
                sim = _cosine_similarity(g["centroid"], det.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_group = g
            if best_group is not None and best_sim >= duplicate_similarity_threshold:
                duplicate_matches += 1
                reject_reasons["duplicate"] += 1
                continue
            if best_group is not None and best_sim >= seed_group_similarity_threshold:
                group_id = int(best_group["id"])
                group_label = str(best_group["label"])
                n = int(best_group["count"]) + 1
                best_group["centroid"] = (best_group["centroid"] * (n - 1) + det.embedding) / n
                norm = float(np.linalg.norm(best_group["centroid"]))
                if norm > 1e-9:
                    best_group["centroid"] = best_group["centroid"] / norm
                best_group["count"] = n
                groups_matched += 1
            else:
                group_id = create_visual_group(
                    conn,
                    production_id=production_id,
                    review_state="pending",
                    expansion_state="blocked",
                    representative_image_path=det.crop_path,
                )
                created = list_visual_groups(conn, production_id=production_id)
                created_group = next((x for x in created if int(x["id"]) == int(group_id)), None)
                group_label = created_group["label"] if created_group else f"visual_person_{group_id:03d}"
                group_centroids.append(
                    {
                        "id": group_id,
                        "label": group_label,
                        "centroid": det.embedding.copy(),
                        "count": 1,
                    }
                )
                groups_created += 1
                best_sim = 1.0

            create_visual_seed(
                conn,
                group_id=group_id,
                track_id=track_id,
                detection_id=det.db_id,
                image_path=det.crop_path,
                embedding=det.embedding.astype(float).tolist(),
                area_ratio=float(det.area_ratio),
                sharpness=float(det.sharpness),
                confidence=float(det.confidence),
                seed_quality_score=quality_score,
                notes=f"auto seed; group_sim={best_sim:.3f}",
            )
            seeds_accepted += 1

        rejected_total = int(sum(reject_reasons.values()))
        verifier_rejects_for_log = int(verifier_rejected_after_quality)

        # Progress reporting every 5 % of the video (or every 50 sampled frames as fallback)
        if total_sample_frames:
            pct = int(sampled_frames * 100 / total_sample_frames)
            pct_milestone = (pct // 5) * 5
            if pct_milestone > _last_pct_reported:
                _last_pct_reported = pct_milestone
                elapsed_s = timestamp_ms / 1000.0
                logger.info(
                    "[%d%%] %.0fs / %.0fs | sampled frames=%d | detections considered=%d | rejected_small=%d | rejected_blurry=%d | rejected_pose=%d | rejected_occluded=%d | rejected_dark=%d | rejected_quality_score=%d | quality_passed_before_verifier=%d | verifier_rejects=%d | duplicate_matches=%d | rejected_total=%d | high_quality_seeds_accepted=%d | new_visual_groups_created=%d | matched_existing_groups=%d",
                    pct_milestone,
                    elapsed_s,
                    duration_s_est or 0.0,
                    sampled_frames,
                    detections_considered,
                    reject_reasons["small"],
                    reject_reasons["blurry"],
                    reject_reasons["pose"],
                    reject_reasons["occluded"],
                    reject_reasons["dark"],
                    reject_reasons["quality_score"],
                    quality_passed_before_verifier,
                    verifier_rejects_for_log,
                    duplicate_matches,
                    rejected_total,
                    seeds_accepted,
                    groups_created,
                    groups_matched,
                )
        elif sampled_frames % 50 == 0:
            elapsed_s = timestamp_ms / 1000.0
            logger.info(
                "[frame %d sampled] %.0fs elapsed | detections=%d | quality_passed_before_verifier=%d | verifier_rejects=%d | duplicate_matches=%d | rejected_total=%d | seeds_accepted=%d",
                sampled_frames,
                elapsed_s,
                detections_considered,
                quality_passed_before_verifier,
                verifier_rejects_for_log,
                duplicate_matches,
                rejected_total,
                seeds_accepted,
            )

    cap.release()
    rebuild_overlay_for_video(conn, video_id)
    set_video_scan_status(conn, video_id, "completed")
    conn.close()

    rejected_total = int(sum(reject_reasons.values()))
    verifier_rejects_for_log = int(verifier_rejected_after_quality)
    stats = {
        "sampled_frames": sampled_frames,
        "detections_considered": detections_considered,
        "accepted_seeds": seeds_accepted,
        "rejected_total": rejected_total,
        "rejected_small": reject_reasons["small"],
        "rejected_blurry": reject_reasons["blurry"],
        "rejected_pose": reject_reasons["pose"],
        "rejected_occluded": reject_reasons["occluded"],
        "rejected_dark": reject_reasons["dark"],
        "rejected_quality_score": reject_reasons["quality_score"],
        "rejected_verifier": reject_reasons["verifier"],
        "duplicate_matches": duplicate_matches,
        "rejected_other": reject_reasons["other"],
        "verifier_rejects": verifier_rejects_for_log,
        "quality_passed_before_verifier": quality_passed_before_verifier,
        "verifier_rejects_after_quality": verifier_rejects_for_log,
        "high_quality_seeds_accepted": seeds_accepted,
        "new_visual_groups_created": groups_created,
        "matched_existing_groups": groups_matched,
    }
    debug_stats_path = None
    if write_debug_stats:
        run_debug_dir = Path(debug_stats_dir).resolve() if debug_stats_dir else (data_root / "debug" / "seed_runs")
        run_debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        debug_stats_path = run_debug_dir / f"{source.stem}_{timestamp_slug}.json"
        debug_payload = {
            "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "video_path": str(source),
            "video_id": video_id,
            "production_id": production_id,
            "thresholds": {
                "sample_fps": sample_fps,
                "start_offset_seconds": start_offset_seconds,
                "max_sampled_frames": max_sampled_frames,
                "min_face_area_ratio": min_face_area_ratio,
                "min_face_size_px": min_face_size_px,
                "max_aspect_ratio_deviation": max_aspect_ratio_deviation,
                "min_sharpness": min_sharpness,
                "min_brightness": min_brightness,
                "min_quality_score": min_quality_score,
                "seed_acceptance_threshold": seed_acceptance_threshold,
                "seed_group_similarity_threshold": seed_group_similarity_threshold,
                "duplicate_similarity_threshold": duplicate_similarity_threshold,
                "verifier_enabled": bool(verifier is not None),
                "verifier_score_threshold": verifier_score_threshold,
                "verifier_min_area_ratio": verifier_min_area_ratio,
                "verifier_max_center_offset": verifier_max_center_offset,
            },
            "stats": stats,
            "runtime": diag_info,
        }
        debug_stats_path.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Seed debug stats written: %s", debug_stats_path)

    logger.info(
        "Seed discovery finished: %s | sampled frames=%d | detections considered=%d | rejected_small=%d | rejected_blurry=%d | rejected_pose=%d | rejected_occluded=%d | rejected_dark=%d | rejected_quality_score=%d | quality_passed_before_verifier=%d | verifier_rejects_after_quality=%d | duplicate_matches=%d | rejected_total=%d | high_quality_seeds_accepted=%d | pseudo_tracks=%d | new_visual_groups_created=%d | matched_existing_groups=%d",
        source.name,
        sampled_frames,
        detections_considered,
        reject_reasons["small"],
        reject_reasons["blurry"],
        reject_reasons["pose"],
        reject_reasons["occluded"],
        reject_reasons["dark"],
        reject_reasons["quality_score"],
        quality_passed_before_verifier,
        verifier_rejects_for_log,
        duplicate_matches,
        rejected_total,
        seeds_accepted,
        pseudo_tracks,
        groups_created,
        groups_matched,
    )

    return {
        "production_id": production_id,
        "video_id": video_id,
        "video_path": str(source),
        "sampled_frames": sampled_frames,
        "detections_considered": detections_considered,
        "rejected_total": rejected_total,
        "rejected_small": reject_reasons["small"],
        "rejected_blurry": reject_reasons["blurry"],
        "rejected_pose": reject_reasons["pose"],
        "rejected_occluded": reject_reasons["occluded"],
        "rejected_dark": reject_reasons["dark"],
        "rejected_quality_score": reject_reasons["quality_score"],
        "rejected_verifier": reject_reasons["verifier"],
        "rejected_duplicate": reject_reasons["duplicate"],
        "rejected_other": reject_reasons["other"],
        "low_quality_rejections": (
            reject_reasons["small"]
            + reject_reasons["blurry"]
            + reject_reasons["pose"]
            + reject_reasons["occluded"]
            + reject_reasons["dark"]
            + reject_reasons["quality_score"]
        ),
        "seeds_accepted": seeds_accepted,
        "verifier_rejections": verifier_rejects_for_log,
        "verifier_rejects_after_quality": verifier_rejects_for_log,
        "quality_passed_before_verifier": quality_passed_before_verifier,
        "duplicate_matches": duplicate_matches,
        "groups_created": groups_created,
        "groups_matched": groups_matched,
        "tracks": pseudo_tracks,
        "clear_tracks": pseudo_tracks,
        "sample_fps": sample_fps,
        "detector_backend": detector_runtime.get("backend"),
        "detector_target": detector_runtime.get("target"),
        "verifier_enabled": bool(verifier is not None),
        "facenet_enabled": bool(facenet_model is not None),
        "facenet_backend": facenet_runtime.get("backend", "pixel_hash_fallback"),
        "gpu_diagnostics": diag_info,
        "seed_debug_stats_path": str(debug_stats_path) if debug_stats_path else None,
    }


def scan_directory(
    directory: str,
    *,
    production: str | None = None,
    skip_done: bool = True,
    recursive: bool = False,
    **scan_kwargs,
) -> dict:
    """Scan all video files found in *directory*.

    Parameters
    ----------
    directory:
        Root folder to search.
    production:
        Override production name. Defaults to the folder name.
    skip_done:
        When True (default), skip videos that already have scan_status='completed'.
    recursive:
        When True, search sub-directories as well.
    **scan_kwargs:
        Forwarded to :func:`scan_video` (fps, thresholds, …).
    """
    from db.database import get_connection as _get_conn, ensure_schema as _ensure_schema  # noqa: F401

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    pattern = "**/*" if recursive else "*"
    video_files = sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in _SCAN_VIDEO_EXTENSIONS
    )

    if not video_files:
        logger.warning("scan_directory: no video files found in %s", root)
        return {"scanned": 0, "skipped": 0, "failed": 0, "results": []}

    production_name = production or root.name

    if skip_done:
        _ensure_schema()
        conn = _get_conn()
        from db.database import list_videos as _list_videos
        existing = {
            v["video_path"]: v["scan_status"]
            for v in _list_videos(conn)
        }
        conn.close()
    else:
        existing = {}

    results: list[dict] = []
    skipped = 0
    failed = 0

    for vf in video_files:
        if skip_done and existing.get(str(vf)) == "completed":
            logger.info("scan_directory: skipping already-completed %s", vf.name)
            skipped += 1
            continue
        logger.info("scan_directory: scanning %s", vf.name)
        try:
            result = scan_video(
                str(vf),
                production=production_name,
                **scan_kwargs,
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("scan_directory: failed on %s – %s", vf.name, exc)
            failed += 1

    return {
        "directory": str(root),
        "production": production_name,
        "scanned": len(results),
        "skipped": skipped,
        "failed": failed,
        "results": results,
    }


def run_expansion_orchestrator(
    *,
    match_threshold: float = 0.70,
) -> dict:
    """Run Step-1C expansion only for explicitly released episodes/groups."""
    ensure_schema()
    conn = get_connection()
    try:
        videos = list_videos(conn)
        released_video_ids = {
            int(v["id"])
            for v in videos
            if bool(v.get("expansion_released")) and str(v.get("scan_status")) == "completed"
        }
        if not released_video_ids:
            logger.info("Expansion orchestrator: no released+completed episodes found.")
            return {"groups_run": 0, "groups_skipped": 0, "released_videos": 0, "results": []}

        groups = list_visual_groups(conn)
        runnable_groups = []
        skipped = 0
        for g in groups:
            if g.get("review_state") in {"ignored", "irrelevant"}:
                skipped += 1
                continue
            if g.get("review_state") != "confirmed":
                skipped += 1
                continue
            if g.get("expansion_state") != "ready":
                skipped += 1
                continue
            prod_id = g.get("production_id")
            if prod_id is None:
                skipped += 1
                continue
            prod_video_ids = {int(v["id"]) for v in videos if v.get("production_id") == prod_id}
            allowed_ids = sorted(prod_video_ids & released_video_ids)
            if not allowed_ids:
                skipped += 1
                continue
            runnable_groups.append((int(g["id"]), allowed_ids))

        results: list[dict] = []
        for group_id, allowed_video_ids in runnable_groups:
            logger.info(
                "Expansion orchestrator: running group_id=%d on %d released episodes",
                group_id,
                len(allowed_video_ids),
            )
            result = run_expansion_for_group(
                conn,
                group_id,
                match_threshold=match_threshold,
                allowed_video_ids=allowed_video_ids,
            )
            result["allowed_video_ids"] = allowed_video_ids
            results.append(result)

        return {
            "groups_run": len(runnable_groups),
            "groups_skipped": skipped,
            "released_videos": len(released_video_ids),
            "results": results,
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step-1 scanner (seed discovery with --video, expansion orchestrator without video)",
        epilog=(
            "Examples:\n"
            "  # Mode A: single episode seed discovery (seed-first)\n"
            "  python scanner.py --video /data/ShowName.S01E02.mkv\n\n"
            "  # Mode A (batch): whole directory seed discovery\n"
            "  python scanner.py --dir /data/ShowName/ --production 'Show Name'\n\n"
            "  # Mode B: expansion orchestrator (only confirmed+ready groups on released episodes)\n"
            "  python scanner.py\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--video",
        help=(
            "Absolute path to a single video file. For spaces: quote the path or "
            "use backslash escapes."
        ),
    )
    mode.add_argument(
        "--dir",
        dest="directory",
        metavar="DIRECTORY",
        help="Scan all video files in this directory.",
    )

    parser.add_argument("--production", help="Production title (series/movie); defaults to parent folder name")
    parser.add_argument("--title", help="Video/episode title (single-video mode only)")
    parser.add_argument("--episode-code", help="Override episode code, e.g. S01E01 (single-video mode only)")
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Force re-scan even for videos that are already status=completed (directory mode only; single-video always rescans)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search sub-directories recursively (directory mode only)",
    )
    parser.add_argument("--fps", type=float, default=float(os.getenv("FACE_SCAN_FPS", "4.0")))
    parser.add_argument("--start-offset-seconds", type=float, default=float(os.getenv("FACE_SCAN_START_OFFSET_SECONDS", "0.0")))
    parser.add_argument("--max-sampled-frames", type=int, default=int(os.getenv("FACE_SCAN_MAX_SAMPLED_FRAMES", "0")))
    parser.add_argument("--min-clear-seconds", type=float, default=float(os.getenv("FACE_MIN_CLEAR_SECONDS", "2.0")))
    parser.add_argument("--min-face-area-ratio", type=float, default=float(os.getenv("FACE_MIN_AREA_RATIO", "0.06")))
    parser.add_argument("--min-sharpness", type=float, default=float(os.getenv("FACE_MIN_SHARPNESS", "70.0")))
    parser.add_argument("--min-brightness", type=float, default=float(os.getenv("FACE_MIN_BRIGHTNESS", "40.0")))
    parser.add_argument("--min-quality-score", type=float, default=float(os.getenv("FACE_MIN_QUALITY_SCORE", "0.55")))
    parser.add_argument("--seed-acceptance-threshold", type=float, default=float(os.getenv("FACE_SEED_ACCEPTANCE_THRESHOLD", "0.60")))
    parser.add_argument("--max-aspect-ratio-deviation", type=float, default=float(os.getenv("FACE_MAX_ASPECT_RATIO_DEVIATION", "0.65")))
    parser.add_argument("--min-stability", type=float, default=float(os.getenv("FACE_MIN_STABILITY", "0.45")))
    parser.add_argument(
        "--dnn-confidence",
        type=float,
        default=float(os.getenv("FACE_DETECTOR_SCORE_THRESHOLD", os.getenv("FACE_DNN_CONFIDENCE", "0.65"))),
        help="Minimum Torch detector confidence 0..1 (higher = fewer false positives, default: 0.65)",
    )
    parser.add_argument("--min-face-size-px", type=int, default=int(os.getenv("FACE_MIN_SIZE_PX", "80")))
    parser.add_argument(
        "--disable-verifier",
        action="store_true",
        help="Disable second-stage face verifier model (not recommended for precision).",
    )
    parser.add_argument(
        "--verifier-score-threshold",
        type=float,
        default=float(os.getenv("FACE_VERIFIER_SCORE_THRESHOLD", "0.92")),
    )
    parser.add_argument(
        "--verifier-min-area-ratio",
        type=float,
        default=float(os.getenv("FACE_VERIFIER_MIN_AREA_RATIO", "0.25")),
    )
    parser.add_argument(
        "--verifier-max-center-offset",
        type=float,
        default=float(os.getenv("FACE_VERIFIER_MAX_CENTER_OFFSET", "0.45")),
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU backend (disables CUDA attempt).",
    )
    parser.add_argument(
        "--detector-device",
        default=os.getenv("FACE_DETECTOR_DEVICE"),
        help="Torch device for face detection (cpu | cuda | cuda:<id>).",
    )
    parser.add_argument(
        "--verifier-device",
        default=os.getenv("FACE_VERIFIER_DEVICE"),
        help="Torch device for verifier (cpu | cuda | cuda:<id>).",
    )
    parser.add_argument(
        "--embedding-device",
        default=os.getenv("FACE_EMBEDDING_DEVICE"),
        help="Torch device for embedding model (cpu | cuda | cuda:<id>).",
    )
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=int(os.getenv("FACE_GPU_DEVICE_ID", "0")),
        help="Legacy fallback CUDA device index when component device env vars are not set.",
    )
    parser.add_argument(
        "--gpu-diagnostics",
        action="store_true",
        default=_env_bool("FACE_GPU_DIAGNOSTICS", True),
        help="Log Torch/CUDA diagnostics at scan startup.",
    )
    parser.add_argument(
        "--diagnose-torch",
        action="store_true",
        help="Print Torch/CUDA diagnostics and exit (no scan).",
    )
    parser.add_argument(
        "--write-debug-stats",
        action="store_true",
        default=_env_bool("FACE_SEED_DEBUG_STATS_ENABLED", False),
        help="Write per-run seed statistics JSON.",
    )
    parser.add_argument(
        "--debug-stats-dir",
        default=os.getenv("FACE_SEED_DEBUG_STATS_DIR"),
        help="Directory for per-run seed debug statistics JSON.",
    )
    parser.add_argument(
        "--diagnose-opencv",
        action="store_true",
        help="Backward-compatible alias for --diagnose-torch.",
    )
    args = parser.parse_args()

    if args.diagnose_torch or args.diagnose_opencv:
        try:
            info = _torch_diagnostics(
                prefer_gpu=_env_bool("FACE_GPU_ENABLED", True) and not args.cpu_only,
                gpu_device_id=args.gpu_device_id,
            )
            logger.info("Torch diagnostics: %s", info)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not collect Torch diagnostics: %s", exc)
            raise SystemExit(2) from exc
        return

    scan_kwargs = dict(
        sample_fps=args.fps,
        start_offset_seconds=args.start_offset_seconds,
        max_sampled_frames=args.max_sampled_frames,
        min_clear_seconds=args.min_clear_seconds,
        min_face_area_ratio=args.min_face_area_ratio,
        min_sharpness=args.min_sharpness,
        min_brightness=args.min_brightness,
        min_quality_score=args.min_quality_score,
        seed_acceptance_threshold=args.seed_acceptance_threshold,
        max_aspect_ratio_deviation=args.max_aspect_ratio_deviation,
        min_stability=args.min_stability,
        dnn_confidence=args.dnn_confidence,
        min_face_size_px=args.min_face_size_px,
        verifier_enabled=_env_bool("FACE_VERIFIER_ENABLED", True) and not args.disable_verifier,
        verifier_score_threshold=args.verifier_score_threshold,
        verifier_min_area_ratio=args.verifier_min_area_ratio,
        verifier_max_center_offset=args.verifier_max_center_offset,
        prefer_gpu=_env_bool("FACE_GPU_ENABLED", True) and not args.cpu_only,
        gpu_device_id=args.gpu_device_id,
        detector_device=args.detector_device,
        verifier_device=args.verifier_device,
        embedding_device=args.embedding_device,
        gpu_diagnostics=args.gpu_diagnostics,
        seed_group_similarity_threshold=float(
            os.getenv("FACE_SEED_GROUP_SIMILARITY_THRESHOLD", os.getenv("FACE_VISUAL_GROUP_ASSIGNMENT_THRESHOLD", "0.90"))
        ),
        duplicate_similarity_threshold=float(os.getenv("FACE_SEED_DUPLICATE_SIMILARITY_THRESHOLD", "0.985")),
        write_debug_stats=args.write_debug_stats,
        debug_stats_dir=args.debug_stats_dir,
    )

    if not args.video and not args.directory:
        result = run_expansion_orchestrator(
            match_threshold=float(os.getenv("FACE_EXPAND_THRESHOLD", "0.70"))
        )
        logger.info(
            "Expansion orchestrator completed: %d groups run, %d groups skipped, %d released videos",
            result["groups_run"],
            result["groups_skipped"],
            result["released_videos"],
        )
    elif args.directory:
        result = scan_directory(
            args.directory,
            production=args.production,
            skip_done=not args.rescan,
            recursive=args.recursive,
            **scan_kwargs,
        )
        logger.info(
            "Directory scan completed: %d scanned, %d skipped, %d failed",
            result["scanned"], result["skipped"], result["failed"],
        )
    else:
        result = scan_video(
            video_path=args.video,
            production=args.production,
            title=args.title,
            episode_code=args.episode_code,
            **scan_kwargs,
        )
        logger.info("Scan completed: %s", result)


if __name__ == "__main__":
    main()
