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


def _parse_production_from_path(path: Path) -> str:
    if path.parent.name:
        return path.parent.name
    return "Unknown Production"


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _build_face_descriptor(crop_bgr: np.ndarray) -> np.ndarray:
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
    min_sharpness: float = 40.0,
    min_stability: float = 0.18,
    track_max_gap_ms: int = 600,
    iou_threshold: float = 0.30,
    descriptor_threshold: float = 0.72,
) -> dict:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV not installed. Install opencv-python-headless.") from exc

    source = Path(video_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    production_name = production or _parse_production_from_path(source)
    video_title = title or source.stem
    ep_code = episode_code or _parse_episode_code(source.name)
    duration_ms = _probe_duration_ms(str(source))

    data_root = Path(os.getenv("FACE_DATA_DIR", "data/faces")).resolve()
    crops_root = data_root / "crops" / source.stem
    reps_root = data_root / "tracks" / source.stem
    crops_root.mkdir(parents=True, exist_ok=True)
    reps_root.mkdir(parents=True, exist_ok=True)

    ensure_schema()
    conn = get_connection()
    production_id, video_id = upsert_production_and_video(
        conn,
        production_title=production_name,
        video_title=video_title,
        video_path=str(source),
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

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        set_video_scan_status(conn, video_id, "failed")
        conn.close()
        raise RuntimeError(f"Could not load face detector cascade: {cascade_path}")

    active_tracks: list[Track] = []
    finished_tracks: list[Track] = []
    frame_index = -1
    sampled_frames = 0
    detection_count = 0

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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(36, 36),
        )

        frame_detections: list[Detection] = []
        for i, (x, y, fw, fh) in enumerate(faces):
            x = int(max(0, x))
            y = int(max(0, y))
            fw = int(min(fw, w - x))
            fh = int(min(fh, h - y))
            if fw <= 0 or fh <= 0:
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
                confidence=0.92,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-1 visual face/track scanner")
    parser.add_argument("--video", required=True, help="Absolute path to source video")
    parser.add_argument("--production", help="Production title (series/movie)")
    parser.add_argument("--title", help="Video/episode title")
    parser.add_argument("--episode-code", help="Optional episode code (e.g. S01E01)")
    parser.add_argument("--fps", type=float, default=float(os.getenv("FACE_SCAN_FPS", "4.0")))
    parser.add_argument("--min-clear-seconds", type=float, default=float(os.getenv("FACE_MIN_CLEAR_SECONDS", "2.0")))
    parser.add_argument("--min-face-area-ratio", type=float, default=float(os.getenv("FACE_MIN_AREA_RATIO", "0.04")))
    parser.add_argument("--min-sharpness", type=float, default=float(os.getenv("FACE_MIN_SHARPNESS", "40.0")))
    parser.add_argument("--min-stability", type=float, default=float(os.getenv("FACE_MIN_STABILITY", "0.18")))
    args = parser.parse_args()

    result = scan_video(
        video_path=args.video,
        production=args.production,
        title=args.title,
        episode_code=args.episode_code,
        sample_fps=args.fps,
        min_clear_seconds=args.min_clear_seconds,
        min_face_area_ratio=args.min_face_area_ratio,
        min_sharpness=args.min_sharpness,
        min_stability=args.min_stability,
    )
    logger.info("Scan completed: %s", result)


if __name__ == "__main__":
    main()
