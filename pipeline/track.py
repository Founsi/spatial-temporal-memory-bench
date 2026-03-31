from __future__ import annotations

"""
Stage 3: Object Tracking

Assigns persistent object IDs across frames by matching objects based on
label similarity and bounding box IoU (Intersection over Union).

This ensures the same real-world object gets the same ID across the sequence,
which is critical for entity resolution evaluation.
"""

from typing import Any

from .config import PipelineConfig


def track_objects(frames: list[dict], config: PipelineConfig) -> list[dict]:
    """
    Assign persistent object IDs across frames.

    Each object in the first frame gets a new ID. In subsequent frames,
    objects are matched to existing IDs using label + IoU matching.
    Unmatched objects get new IDs.

    Returns frames with updated object_id fields.
    """
    print(f"  Tracking objects across {len(frames)} frames...")

    tracked_objects: dict[str, dict] = {}  # object_id -> latest object info
    next_id_counter = 1
    tracked_frames = []

    for i, frame in enumerate(frames):
        objects = frame.get("objects", [])
        updated_objects = []

        # Match each detected object to existing tracked objects
        unmatched_tracked = set(tracked_objects.keys())
        assignments: list[tuple[int, str, float]] = []  # (obj_idx, tracked_id, score)

        for obj_idx, obj in enumerate(objects):
            best_match_id = None
            best_score = 0.0

            for tracked_id, tracked_info in tracked_objects.items():
                score = _match_score(obj, tracked_info, config)
                if score > best_score:
                    best_score = score
                    best_match_id = tracked_id

            if best_match_id and best_score > 0:
                assignments.append((obj_idx, best_match_id, best_score))

        # Resolve conflicts: if multiple objects match the same tracked ID,
        # keep the best match
        assignments.sort(key=lambda x: x[2], reverse=True)
        used_obj_indices = set()
        used_tracked_ids = set()

        for obj_idx, tracked_id, score in assignments:
            if obj_idx in used_obj_indices or tracked_id in used_tracked_ids:
                continue
            used_obj_indices.add(obj_idx)
            used_tracked_ids.add(tracked_id)

        # Build updated objects list
        for obj_idx, obj in enumerate(objects):
            matched_id = None
            for a_idx, a_tid, a_score in assignments:
                if a_idx == obj_idx and a_idx in used_obj_indices and a_tid in used_tracked_ids:
                    # Check this was the winning assignment
                    if a_idx in used_obj_indices:
                        matched_id = a_tid
                    break

            if matched_id:
                obj_id = matched_id
                unmatched_tracked.discard(matched_id)
            else:
                obj_id = f"obj_{_sanitize_label(obj.get('label', 'unknown'))}_{next_id_counter:02d}"
                next_id_counter += 1

            updated_obj = {**obj, "object_id": obj_id}
            updated_objects.append(updated_obj)

            # Update tracking state
            tracked_objects[obj_id] = {
                "label": obj.get("label", ""),
                "bounding_box": obj.get("bounding_box", {}),
                "state": obj.get("state", ""),
                "last_seen_frame": i,
            }

        tracked_frame = {**frame, "objects": updated_objects}
        tracked_frames.append(tracked_frame)

    # Report tracking stats
    total_objects = len(tracked_objects)
    objects_per_frame = [len(f.get("objects", [])) for f in tracked_frames]
    print(f"  Tracked {total_objects} unique objects")
    print(f"  Objects per frame: {objects_per_frame}")

    return tracked_frames


def _match_score(
    candidate: dict, tracked: dict, config: PipelineConfig
) -> float:
    """
    Compute a match score between a candidate object and a tracked object.
    Returns 0 if no match, positive float if match (higher = better).
    """
    # Label matching
    cand_label = candidate.get("label", "").lower().strip()
    track_label = tracked.get("label", "").lower().strip()

    if config.label_match_required:
        if not _labels_similar(cand_label, track_label):
            return 0.0
        label_score = 1.0
    else:
        label_score = 1.0 if _labels_similar(cand_label, track_label) else 0.3

    # IoU matching
    cand_box = candidate.get("bounding_box", {})
    track_box = tracked.get("bounding_box", {})

    if cand_box and track_box:
        iou = _compute_iou(cand_box, track_box)
        if iou < config.iou_threshold:
            # Low IoU — object might have moved. Still allow match if labels match well,
            # since objects DO move between frames (that's what we're testing).
            spatial_score = 0.2
        else:
            spatial_score = iou
    else:
        spatial_score = 0.5  # No box info, rely on label

    return label_score * 0.6 + spatial_score * 0.4


def _labels_similar(a: str, b: str) -> bool:
    """Check if two object labels refer to the same thing."""
    if a == b:
        return True
    # Handle common variations
    a_parts = set(a.replace("_", " ").split())
    b_parts = set(b.replace("_", " ").split())
    # If significant word overlap, consider similar
    if len(a_parts & b_parts) >= max(1, min(len(a_parts), len(b_parts)) - 1):
        return True
    return False


def _compute_iou(box_a: dict, box_b: dict) -> float:
    """Compute Intersection over Union for two bounding boxes."""
    x_min_a = box_a.get("x_min", 0)
    y_min_a = box_a.get("y_min", 0)
    x_max_a = box_a.get("x_max", 0)
    y_max_a = box_a.get("y_max", 0)

    x_min_b = box_b.get("x_min", 0)
    y_min_b = box_b.get("y_min", 0)
    x_max_b = box_b.get("x_max", 0)
    y_max_b = box_b.get("y_max", 0)

    # Intersection
    inter_x_min = max(x_min_a, x_min_b)
    inter_y_min = max(y_min_a, y_min_b)
    inter_x_max = min(x_max_a, x_max_b)
    inter_y_max = min(y_max_a, y_max_b)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Union
    area_a = (x_max_a - x_min_a) * (y_max_a - y_min_a)
    area_b = (x_max_b - x_min_b) * (y_max_b - y_min_b)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _sanitize_label(label: str) -> str:
    """Convert a label to a safe ID component."""
    return label.lower().replace(" ", "_").replace("-", "_")[:20]
