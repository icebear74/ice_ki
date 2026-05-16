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
import logging
import os
import re
import shutil
import urllib.request
from dataclasses import dataclass, field
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
    ensure_schema,
    get_connection,
    rebuild_overlay_for_video,
    set_video_scan_status,
    upsert_production_and_video,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


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


# ---------------------------------------------------------------------------
# DNN face detector (OpenCV ResNet-SSD, much more accurate than Haar cascade)
# ---------------------------------------------------------------------------
_DNN_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/"
    "samples/dnn/face_detector/deploy.prototxt"
)
_DNN_MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20180205_fp16/"
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)
_DNN_PROTOTXT_NAME = "face_deploy.prototxt"
_DNN_MODEL_NAME = "res10_300x300_ssd_fp16.caffemodel"


def _get_dnn_face_detector(models_dir: Path):
    """Load (downloading once) the OpenCV ResNet-SSD DNN face detector.

    Tries to use the CUDA backend (GPU) first; falls back to CPU if not
    available (standard opencv-python-headless has no CUDA support).
    Returns the ``cv2.dnn_Net`` object.
    """
    import cv2

    models_dir.mkdir(parents=True, exist_ok=True)
    prototxt = models_dir / _DNN_PROTOTXT_NAME
    caffemodel = models_dir / _DNN_MODEL_NAME

    if not prototxt.exists():
        logger.info("Downloading DNN face-detector prototxt to %s …", prototxt)
        try:
            urllib.request.urlretrieve(_DNN_PROTOTXT_URL, prototxt)
        except Exception as exc:
            raise RuntimeError(f"Could not download DNN prototxt: {exc}") from exc

    if not caffemodel.exists():
        logger.info(
            "Downloading DNN face-detector model (~2 MB) to %s …", caffemodel
        )
        try:
            urllib.request.urlretrieve(_DNN_MODEL_URL, caffemodel)
        except Exception as exc:
            raise RuntimeError(f"Could not download DNN model: {exc}") from exc

    net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))

    # Attempt GPU (CUDA) backend – only works if OpenCV was compiled with CUDA.
    gpu_ok = False
    try:
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            gpu_ok = True
    except Exception:
        pass

    if gpu_ok:
        logger.info("Face detector: CUDA backend active (GPU accelerated ✓)")
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info(
            "Face detector: CPU backend "
            "(for GPU: install opencv-python with CUDA support)"
        )

    return net


def _dnn_detect_faces(
    net,
    frame: np.ndarray,
    min_confidence: float,
    min_size_px: int,
) -> list[tuple[int, int, int, int, float]]:
    """Run the DNN face detector on *frame*.

    Returns a list of ``(x, y, w, h, confidence)`` tuples for every detected
    face that passes the confidence and minimum-size filters.
    """
    import cv2

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    raw = net.forward()  # shape (1, 1, N, 7)

    results: list[tuple[int, int, int, int, float]] = []
    for i in range(raw.shape[2]):
        confidence = float(raw[0, 0, i, 2])
        if confidence < min_confidence:
            continue
        box = raw[0, 0, i, 3:7] * np.array([w, h, w, h], dtype=np.float32)
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        fw, fh = x2 - x1, y2 - y1
        if fw < min_size_px or fh < min_size_px:
            continue
        results.append((x1, y1, fw, fh, confidence))
    return results


    import cv2

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 8), interpolation=cv2.INTER_AREA)
    desc = small.astype(np.float32).reshape(-1) / 255.0
    norm = np.linalg.norm(desc)
    if norm > 1e-9:
        desc = desc / norm
    return desc


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
    dnn_confidence: float = 0.50,
    min_face_size_px: int = 60,
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

    # Remove stale image data on every (re-)scan so the disk stays in sync with the DB.
    for stale_dir in (crops_root, reps_root):
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
            logger.info("Removed stale scan images: %s", stale_dir)

    crops_root.mkdir(parents=True, exist_ok=True)
    reps_root.mkdir(parents=True, exist_ok=True)

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
        video_meta={"sample_fps": sample_fps},
    )

    logger.info("Scanning video_id=%s (%s)", video_id, source)
    set_video_scan_status(conn, video_id, "scanning")
    clear_video_scan_data(conn, video_id)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        set_video_scan_status(conn, video_id, "failed")
        conn.close()
        raise RuntimeError(f"Could not open video: {source}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0:
        native_fps = 25.0
    frame_step = max(1, int(round(native_fps / max(sample_fps, 0.5))))

    # Load DNN face detector (downloads model once on first run).
    models_dir = data_root / "models"
    try:
        face_net = _get_dnn_face_detector(models_dir)
    except Exception as exc:
        set_video_scan_status(conn, video_id, "failed")
        conn.close()
        cap.release()
        raise RuntimeError(f"Could not load DNN face detector: {exc}") from exc

    active_tracks: list[Track] = []
    finished_tracks: list[Track] = []
    frame_index = -1
    sampled_frames = 0
    detection_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_sample_frames = max(1, total_frames // frame_step) if total_frames > 0 else None
    duration_s_est = (total_frames / native_fps) if (native_fps > 0 and total_frames > 0) else None
    _last_pct_reported = -1  # tracks last reported percentage milestone

    if duration_s_est:
        logger.info(
            "Starting scan: %s | duration ~%.0fs | ~%d frames to sample at %.1f fps",
            source.name, duration_s_est, total_sample_frames or 0, sample_fps,
        )
    else:
        logger.info("Starting scan: %s | sampling at %.1f fps", source.name, sample_fps)

    while True:
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

        raw_faces = _dnn_detect_faces(face_net, frame, dnn_confidence, min_face_size_px)

        frame_detections: list[Detection] = []
        for i, (x, y, fw, fh, conf) in enumerate(raw_faces):
            fw = int(min(fw, w - x))
            fh = int(min(fh, h - y))
            if fw <= 0 or fh <= 0:
                continue
            # Skip detections with extreme aspect ratios – real faces are roughly square.
            aspect = fw / fh
            if aspect < 0.5 or aspect > 2.0:
                continue
            crop = frame[y : y + fh, x : x + fw]
            if crop.size == 0:
                continue
            sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            area_ratio = (fw * fh) / frame_area
            descriptor = _build_face_descriptor(crop)

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
                metadata={"area_ratio": area_ratio},
            )
            frame_detections.append(det)
            detection_count += 1

        for track in list(active_tracks):
            if timestamp_ms - track.last_ts_ms > track_max_gap_ms:
                finished_tracks.append(track)
                active_tracks.remove(track)

        unmatched = frame_detections[:]
        for det in frame_detections:
            best_track = None
            best_score = -1.0
            for track in active_tracks:
                iou = _iou(track.last_bbox, det.bbox)
                sim = _cosine_similarity(track.detections[-1].embedding, det.embedding)
                score = 0.7 * iou + 0.3 * max(0.0, sim)
                if iou >= iou_threshold and sim >= descriptor_threshold and score > best_score:
                    best_score = score
                    best_track = track
            if best_track is not None:
                best_track.append(det)
                if det in unmatched:
                    unmatched.remove(det)

        for det in unmatched:
            t = Track()
            t.append(det)
            active_tracks.append(t)

        # Progress reporting every 5 % of the video (or every 50 sampled frames as fallback)
        if total_sample_frames:
            pct = int(sampled_frames * 100 / total_sample_frames)
            pct_milestone = (pct // 5) * 5
            if pct_milestone > _last_pct_reported:
                _last_pct_reported = pct_milestone
                elapsed_s = timestamp_ms / 1000.0
                logger.info(
                    "[%d%%] %.0fs / %.0fs | faces detected so far: %d | active tracks: %d | finished tracks: %d",
                    pct_milestone,
                    elapsed_s,
                    duration_s_est or 0.0,
                    detection_count,
                    len(active_tracks),
                    len(finished_tracks),
                )
        elif sampled_frames % 50 == 0:
            elapsed_s = timestamp_ms / 1000.0
            logger.info(
                "[frame %d sampled] %.0fs elapsed | faces detected: %d | active tracks: %d",
                sampled_frames, elapsed_s, detection_count, len(active_tracks),
            )

    cap.release()
    finished_tracks.extend(active_tracks)

    persisted_tracks = 0
    clear_tracks = 0

    for track_idx, track in enumerate(finished_tracks, start=1):
        if not track.detections:
            continue
        duration_s = (track.detections[-1].timestamp_ms - track.detections[0].timestamp_ms) / 1000.0
        mean_area = float(np.mean([d.area_ratio for d in track.detections]))
        mean_sharpness_val = float(np.mean([d.sharpness for d in track.detections]))
        mean_conf = float(np.mean([d.confidence for d in track.detections]))
        frame_count = len(track.detections)

        ious = []
        for i in range(1, frame_count):
            ious.append(_iou(track.detections[i - 1].bbox, track.detections[i].bbox))
        stability = float(np.mean(ious)) if ious else 0.0

        quality_score = (
            0.35 * _clamp(mean_area / max(min_face_area_ratio, 1e-6), 0.0, 1.0)
            + 0.35 * _clamp(mean_sharpness_val / max(min_sharpness, 1.0), 0.0, 1.0)
            + 0.30 * _clamp(stability / max(min_stability, 1e-6), 0.0, 1.0)
        )
        relevance_score = (
            0.50 * _clamp(duration_s / max(min_clear_seconds, 1e-6), 0.0, 1.0)
            + 0.20 * _clamp(frame_count / max(sample_fps * min_clear_seconds, 1.0), 0.0, 1.0)
            + 0.30 * _clamp(mean_area / max(min_face_area_ratio, 1e-6), 0.0, 1.0)
        )
        is_clear = (
            duration_s >= min_clear_seconds
            and mean_area >= min_face_area_ratio
            and mean_sharpness_val >= min_sharpness
            and stability >= min_stability
            and frame_count >= int(sample_fps * min_clear_seconds)
        )
        status = "candidate" if is_clear else "background"
        if is_clear:
            clear_tracks += 1

        best_det = max(
            track.detections,
            key=lambda d: (d.sharpness * (0.5 + d.area_ratio)),
        )
        rep_src = data_root / best_det.crop_path
        rep_name = f"track_{track_idx:05d}.jpg"
        rep_path = reps_root / rep_name
        try:
            rep_img = cv2.imread(str(rep_src))
            if rep_img is not None:
                cv2.imwrite(str(rep_path), rep_img)
                rep_rel = _safe_relpath(rep_path, data_root)
            else:
                rep_rel = best_det.crop_path
        except Exception:
            rep_rel = best_det.crop_path

        track_embedding = np.mean(np.array([d.embedding for d in track.detections]), axis=0)
        track_id = create_face_track(
            conn,
            video_id=video_id,
            start_ms=track.detections[0].timestamp_ms,
            end_ms=track.detections[-1].timestamp_ms,
            frame_count=frame_count,
            mean_face_area=mean_area,
            mean_sharpness=mean_sharpness_val,
            mean_confidence=mean_conf,
            stability_score=stability,
            quality_score=quality_score,
            relevance_score=relevance_score,
            is_clear=is_clear,
            status=status,
            representative_image_path=rep_rel,
            embedding=track_embedding.astype(float).tolist(),
            metadata={
                "duration_seconds": duration_s,
                "global_rule": "step1_clear_track_rule",
                "thresholds": {
                    "min_clear_seconds": min_clear_seconds,
                    "min_face_area_ratio": min_face_area_ratio,
                    "min_sharpness": min_sharpness,
                    "min_stability": min_stability,
                },
            },
        )
        persisted_tracks += 1
        for det in track.detections:
            if det.db_id is not None:
                assign_detection_to_track(conn, det.db_id, track_id)

    rebuild_overlay_for_video(conn, video_id)
    set_video_scan_status(conn, video_id, "completed")
    conn.close()

    logger.info(
        "Scan finished: %s | sampled %d frames | %d face detections | %d tracks (%d clear / %d background)",
        source.name,
        sampled_frames,
        detection_count,
        persisted_tracks,
        clear_tracks,
        persisted_tracks - clear_tracks,
    )

    return {
        "production_id": production_id,
        "video_id": video_id,
        "video_path": str(source),
        "sampled_frames": sampled_frames,
        "detections": detection_count,
        "tracks": persisted_tracks,
        "clear_tracks": clear_tracks,
        "sample_fps": sample_fps,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step-1 visual face/track scanner",
        epilog=(
            "Examples:\n"
            "  # Single video (re-scan always replaces old data)\n"
            "  python scanner.py --video /data/ShowName.S01E02.mkv\n\n"
            "  # Whole directory (skip already-completed videos)\n"
            "  python scanner.py --dir /data/ShowName/ --production 'Show Name'\n\n"
            "  # Whole directory, force re-scan of completed videos too\n"
            "  python scanner.py --dir /data/ShowName/ --rescan\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
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
    parser.add_argument("--min-clear-seconds", type=float, default=float(os.getenv("FACE_MIN_CLEAR_SECONDS", "2.0")))
    parser.add_argument("--min-face-area-ratio", type=float, default=float(os.getenv("FACE_MIN_AREA_RATIO", "0.04")))
    parser.add_argument("--min-sharpness", type=float, default=float(os.getenv("FACE_MIN_SHARPNESS", "70.0")))
    parser.add_argument("--min-stability", type=float, default=float(os.getenv("FACE_MIN_STABILITY", "0.30")))
    parser.add_argument(
        "--dnn-confidence",
        type=float,
        default=float(os.getenv("FACE_DNN_CONFIDENCE", "0.50")),
        help="Minimum DNN detection confidence 0..1 (higher = fewer false positives, default: 0.50)",
    )
    parser.add_argument("--min-face-size-px", type=int, default=int(os.getenv("FACE_MIN_SIZE_PX", "60")))
    args = parser.parse_args()

    scan_kwargs = dict(
        sample_fps=args.fps,
        min_clear_seconds=args.min_clear_seconds,
        min_face_area_ratio=args.min_face_area_ratio,
        min_sharpness=args.min_sharpness,
        min_stability=args.min_stability,
        dnn_confidence=args.dnn_confidence,
        min_face_size_px=args.min_face_size_px,
    )

    if args.directory:
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
