from __future__ import annotations

"""
Stage 1: Frame Extraction

Extracts keyframes from video files or loads image sequences from directories.
Supports fixed-interval extraction and scene-change detection.
"""

import hashlib
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .config import PipelineConfig


def extract_frames(config: PipelineConfig) -> list[dict]:
    """
    Extract frames from the configured input source.

    Returns a list of frame dicts:
        [{"frame_id": str, "timestamp": str, "image_path": str}, ...]
    """
    if config.source_type == "video":
        return _extract_from_video(config)
    elif config.source_type == "image_dir":
        return _extract_from_image_dir(config)
    else:
        raise ValueError(
            f"Source type '{config.source_type}' not yet supported in extract. "
            f"Use 'video' or 'image_dir'."
        )


def _extract_from_video(config: PipelineConfig) -> list[dict]:
    """Extract frames from a video file using OpenCV."""
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for video extraction. "
            "Install with: pip install opencv-python"
        )

    video_path = config.input_path
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print(f"  Video: {video_path.name}")
    print(f"  FPS: {fps:.1f}, Duration: {duration_sec:.1f}s, Total frames: {total_frames}")

    output_dir = config.image_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    base_time = datetime.now(timezone.utc).replace(microsecond=0)

    if config.use_scene_change:
        frames = _extract_scene_changes(cap, config, base_time, output_dir)
    else:
        frames = _extract_fixed_interval(cap, config, fps, base_time, output_dir)

    cap.release()

    # Enforce max_frames limit
    if len(frames) > config.max_frames:
        # Keep evenly spaced subset
        step = len(frames) / config.max_frames
        frames = [frames[int(i * step)] for i in range(config.max_frames)]

    print(f"  Extracted {len(frames)} frames")
    return frames


def _extract_fixed_interval(
    cap, config: PipelineConfig, fps: float,
    base_time: datetime, output_dir: Path
) -> list[dict]:
    """Extract frames at fixed time intervals."""
    import cv2

    frame_interval_frames = int(fps * config.frame_interval_sec)
    if frame_interval_frames < 1:
        frame_interval_frames = 1

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval_frames == 0:
            timestamp_sec = frame_idx / fps
            frame_data = _save_frame(
                frame, frame_idx, timestamp_sec, base_time, output_dir, config
            )
            frames.append(frame_data)

        frame_idx += 1

    return frames


def _extract_scene_changes(
    cap, config: PipelineConfig,
    base_time: datetime, output_dir: Path
) -> list[dict]:
    """Extract frames at scene change boundaries using histogram difference."""
    import cv2

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    prev_hist = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)

        if prev_hist is None:
            # Always include first frame
            timestamp_sec = frame_idx / fps
            frame_data = _save_frame(
                frame, frame_idx, timestamp_sec, base_time, output_dir, config
            )
            frames.append(frame_data)
        else:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
            if diff > config.scene_change_threshold:
                timestamp_sec = frame_idx / fps
                frame_data = _save_frame(
                    frame, frame_idx, timestamp_sec, base_time, output_dir, config
                )
                frames.append(frame_data)

        prev_hist = hist
        frame_idx += 1

    return frames


def _save_frame(
    frame, frame_idx: int, timestamp_sec: float,
    base_time: datetime, output_dir: Path, config: PipelineConfig
) -> dict:
    """Save a frame to disk and return its metadata dict."""
    import cv2

    frame_num = len(list(output_dir.glob("frame_*.jpg")))
    frame_id = f"f{frame_num + 1:03d}"
    filename = f"frame_{frame_num + 1:04d}.jpg"
    filepath = output_dir / filename

    cv2.imwrite(str(filepath), frame)

    timestamp = base_time + timedelta(seconds=timestamp_sec)

    return {
        "frame_id": frame_id,
        "timestamp": timestamp.isoformat(),
        "image_path": str(filepath.relative_to(config.output_dir.parent if config.output_dir.parent.exists() else filepath.parent.parent)),
    }


def _extract_from_image_dir(config: PipelineConfig) -> list[dict]:
    """Load frames from a directory of images, sorted by name."""
    image_dir = config.input_path
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Expected directory: {image_dir}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in extensions
    )

    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    output_dir = config.image_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_time = datetime.now(timezone.utc).replace(microsecond=0)
    frames = []

    for i, img_file in enumerate(image_files[:config.max_frames]):
        frame_id = f"f{i + 1:03d}"
        dest = output_dir / f"frame_{i + 1:04d}{img_file.suffix}"
        shutil.copy2(img_file, dest)

        timestamp = base_time + timedelta(seconds=i * config.frame_interval_sec)

        frames.append({
            "frame_id": frame_id,
            "timestamp": timestamp.isoformat(),
            "image_path": str(dest.relative_to(config.output_dir.parent if config.output_dir.parent.exists() else dest.parent.parent)),
        })

    print(f"  Loaded {len(frames)} images from {image_dir}")
    return frames
