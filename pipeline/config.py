"""
Pipeline configuration and defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuration for the curation pipeline."""

    # --- Input ---
    input_path: Path = Path(".")
    source_type: str = "video"  # "video", "image_dir", "ego4d", "charades"
    environment: str = "unknown"

    # --- Frame extraction ---
    max_frames: int = 20
    frame_interval_sec: float = 30.0  # extract a frame every N seconds
    scene_change_threshold: float = 30.0  # threshold for scene-change detection
    use_scene_change: bool = False  # use scene-change detection instead of fixed interval

    # --- Scene analysis ---
    vision_provider: str = "openai"  # "openai", "anthropic", "local"
    vision_model: str = "gpt-4o"  # or "claude-sonnet-4-20250514", or local model name
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # --- Object tracking ---
    iou_threshold: float = 0.3  # IoU threshold for matching objects across frames
    label_match_required: bool = True  # require label match for object tracking

    # --- QA generation ---
    qa_provider: str = "anthropic"  # LLM for generating QA pairs
    qa_model: str = "claude-sonnet-4-20250514"
    questions_per_sample: int = 8
    difficulty_distribution: dict = field(
        default_factory=lambda: {"easy": 0.3, "medium": 0.4, "hard": 0.3}
    )
    target_dimensions: list[str] = field(
        default_factory=lambda: [
            "spatial_recall",
            "temporal_tracking",
            "entity_resolution",
            "contextual_retrieval",
        ]
    )

    # --- Output ---
    output_dir: Path = Path("data/curated")
    image_output_dir: Path = Path("data/curated/images")
    schema_version: str = "0.1.0"
    annotator: str = "pipeline-auto"


# Supported source dataset adapters
SOURCE_ADAPTERS = {
    "video": "pipeline.sources.video",
    "image_dir": "pipeline.sources.image_dir",
    "ego4d": "pipeline.sources.ego4d",
    "charades": "pipeline.sources.charades",
}
