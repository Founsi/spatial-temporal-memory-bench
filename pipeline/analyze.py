from __future__ import annotations

"""
Stage 2: Scene Analysis

Uses vision-language models to detect objects, estimate bounding boxes,
describe object states, and generate scene descriptions for each frame.
"""

import base64
import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig

ANALYSIS_PROMPT = """Analyze this image for a spatial-temporal memory benchmark. Return a JSON object with:

1. "objects": a list of prominent, trackable objects in the scene. For each object:
   - "label": a descriptive, unique label (e.g., "red_mug", "silver_laptop", "blue_backpack")
   - "bounding_box": estimated normalized coordinates {"x_min", "y_min", "x_max", "y_max"} where 0,0 is top-left and 1,1 is bottom-right
   - "state": current state description (e.g., "on_table", "in_hand", "open", "closed", "hanging_on_hook")
   - "spatial_description": natural language description of the object's position in the scene

2. "scene_description": a detailed natural language description of the full scene, mentioning key objects, their arrangement, lighting, and any notable context.

Focus on objects that:
- Could plausibly move or change state over time
- Are distinctive enough to track across frames
- Would be useful for "where did I leave X?" type questions

Return ONLY valid JSON, no markdown formatting or extra text.

Example output:
{
  "objects": [
    {
      "label": "red_mug",
      "bounding_box": {"x_min": 0.45, "y_min": 0.60, "x_max": 0.52, "y_max": 0.72},
      "state": "on_counter",
      "spatial_description": "Red mug sitting on the kitchen counter near the coffee maker"
    }
  ],
  "scene_description": "Morning kitchen scene with natural light. A red mug sits on the counter beside a coffee maker."
}"""


def analyze_frames(
    frames: list[dict], config: PipelineConfig
) -> list[dict]:
    """
    Analyze each frame using a vision model to detect objects and describe scenes.

    Enriches each frame dict with 'objects' and 'scene_description' fields.
    Returns the enriched frames list.
    """
    print(f"  Analyzing {len(frames)} frames with {config.vision_provider}/{config.vision_model}...")

    analyzed = []
    for i, frame in enumerate(frames):
        print(f"  Frame {i + 1}/{len(frames)}: {frame['frame_id']}")
        image_path = _resolve_image_path(frame["image_path"], config)

        try:
            analysis = _analyze_single_frame(image_path, config)
            frame_enriched = {**frame}
            frame_enriched["objects"] = analysis.get("objects", [])
            frame_enriched["scene_description"] = analysis.get("scene_description", "")
            analyzed.append(frame_enriched)
        except Exception as e:
            print(f"    WARNING: Analysis failed for {frame['frame_id']}: {e}")
            analyzed.append({**frame, "objects": [], "scene_description": ""})

    return analyzed


def _resolve_image_path(image_path: str, config: PipelineConfig) -> Path:
    """Resolve relative image path to absolute."""
    p = Path(image_path)
    if p.is_absolute() and p.exists():
        return p
    # Try relative to project root
    project_root = Path(__file__).parent.parent
    candidate = project_root / image_path
    if candidate.exists():
        return candidate
    # Try relative to data dir
    candidate = project_root / "data" / image_path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Image not found: {image_path}")


def _analyze_single_frame(image_path: Path, config: PipelineConfig) -> dict:
    """Send a single frame to the vision model for analysis."""
    if config.vision_provider == "openai":
        return _analyze_with_openai(image_path, config)
    elif config.vision_provider == "anthropic":
        return _analyze_with_anthropic(image_path, config)
    elif config.vision_provider == "local":
        return _analyze_local_placeholder(image_path, config)
    else:
        raise ValueError(f"Unknown vision provider: {config.vision_provider}")


def _encode_image(image_path: Path) -> tuple[str, str]:
    """Read and base64-encode an image. Returns (base64_data, media_type)."""
    suffix = image_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    media_type = media_types.get(suffix, "image/jpeg")

    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def _parse_json_response(text: str) -> dict:
    """Extract JSON from a model response, handling markdown code blocks."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def _analyze_with_openai(image_path: Path, config: PipelineConfig) -> dict:
    """Analyze frame using OpenAI's vision API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")

    client = OpenAI(api_key=config.openai_api_key or None)
    image_data, media_type = _encode_image(image_path)

    response = client.chat.completions.create(
        model=config.vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ANALYSIS_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=2000,
        temperature=0.1,
    )

    return _parse_json_response(response.choices[0].message.content)


def _analyze_with_anthropic(image_path: Path, config: PipelineConfig) -> dict:
    """Analyze frame using Anthropic's vision API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required. Install with: pip install anthropic")

    client = anthropic.Anthropic(api_key=config.anthropic_api_key or None)
    image_data, media_type = _encode_image(image_path)

    response = client.messages.create(
        model=config.vision_model,
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": ANALYSIS_PROMPT},
                ],
            }
        ],
    )

    return _parse_json_response(response.content[0].text)


def _analyze_local_placeholder(image_path: Path, config: PipelineConfig) -> dict:
    """
    Placeholder for local model analysis (e.g., LLaVA, CogVLM).
    Returns a minimal structure for testing the pipeline without API calls.
    """
    return {
        "objects": [
            {
                "label": f"object_from_{image_path.stem}",
                "bounding_box": {"x_min": 0.3, "y_min": 0.3, "x_max": 0.7, "y_max": 0.7},
                "state": "detected",
                "spatial_description": f"Object detected in {image_path.name}",
            }
        ],
        "scene_description": f"Scene from {image_path.name} (local placeholder analysis)",
    }
