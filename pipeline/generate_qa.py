from __future__ import annotations

"""
Stage 4: QA Generation

Auto-generates memory evaluation questions from annotated frame sequences.
Uses an LLM to create diverse, high-quality QA pairs across evaluation dimensions,
with a rule-based fallback for testing without API access.
"""

import json
import random
from typing import Any

from .config import PipelineConfig

QA_GENERATION_PROMPT = """You are generating evaluation questions for a spatial-temporal memory benchmark.

Given the following sequence of annotated frames from a {environment} environment, generate {num_questions} question-answer pairs that test a memory system's ability to recall spatial and temporal information.

## Frame Sequence

{frame_descriptions}

## Requirements

Generate questions across these evaluation dimensions:
{dimensions}

For each question, provide:
- "question_id": sequential ID like "q001", "q002", etc.
- "question": natural language question that requires remembering information from the frames
- "answer": ground-truth answer based on the frame data
- "answer_type": one of "exact", "fuzzy", "multiple_choice", "open_ended"
- "dimensions": list of dimensions this question tests (from the list above)
- "difficulty": "easy", "medium", or "hard"
- "relevant_frames": list of frame_ids needed to answer the question

## Difficulty Guidelines
- easy: Single frame lookup (e.g., "Where was X at time T?")
- medium: Requires comparing 2-3 frames (e.g., "Did X move between T1 and T2?")
- hard: Requires reasoning across many frames, inferring events, or tracking multiple objects (e.g., "What changed in the room over the full sequence?")

## Important
- Questions MUST require memory of previous frames — they should NOT be answerable from a single current frame alone (except easy questions about specific past moments)
- Include questions about objects that disappeared or changed state
- Include at least one question about spatial relationships between objects
- Vary the answer types across questions

Return ONLY a JSON array of question objects, no markdown or extra text.
"""


def generate_qa_pairs(
    frames: list[dict], config: PipelineConfig
) -> list[dict]:
    """
    Generate QA pairs from annotated frames.

    Uses an LLM if configured, falls back to rule-based generation.
    """
    num_questions = config.questions_per_sample

    if config.qa_provider in ("openai", "anthropic"):
        try:
            return _generate_with_llm(frames, config)
        except Exception as e:
            print(f"  WARNING: LLM QA generation failed ({e}), using rule-based fallback")

    return _generate_rule_based(frames, config)


def _build_frame_descriptions(frames: list[dict]) -> str:
    """Build a text description of frames for the LLM prompt."""
    parts = []
    for frame in frames:
        desc = f"### {frame['frame_id']} (timestamp: {frame['timestamp']})\n"
        desc += f"Scene: {frame.get('scene_description', 'No description')}\n"
        objects = frame.get("objects", [])
        if objects:
            desc += "Objects:\n"
            for obj in objects:
                desc += (
                    f"  - {obj.get('object_id', '?')}: {obj.get('label', '?')} "
                    f"[state: {obj.get('state', '?')}] "
                    f"({obj.get('spatial_description', 'no position info')})\n"
                )
        else:
            desc += "Objects: none detected\n"
        parts.append(desc)
    return "\n".join(parts)


def _generate_with_llm(frames: list[dict], config: PipelineConfig) -> list[dict]:
    """Generate QA pairs using an LLM."""
    frame_descriptions = _build_frame_descriptions(frames)
    dimensions = ", ".join(config.target_dimensions)

    prompt = QA_GENERATION_PROMPT.format(
        environment=config.environment,
        num_questions=config.questions_per_sample,
        frame_descriptions=frame_descriptions,
        dimensions=dimensions,
    )

    if config.qa_provider == "anthropic":
        return _generate_anthropic(prompt, config)
    elif config.qa_provider == "openai":
        return _generate_openai(prompt, config)
    else:
        raise ValueError(f"Unknown QA provider: {config.qa_provider}")


def _generate_anthropic(prompt: str, config: PipelineConfig) -> list[dict]:
    """Generate QA using Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required. Install with: pip install anthropic")

    client = anthropic.Anthropic(api_key=config.anthropic_api_key or None)
    response = client.messages.create(
        model=config.qa_model,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    return _parse_qa_response(text)


def _generate_openai(prompt: str, config: PipelineConfig) -> list[dict]:
    """Generate QA using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")

    client = OpenAI(api_key=config.openai_api_key or None)
    response = client.chat.completions.create(
        model=config.qa_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.3,
    )

    text = response.choices[0].message.content
    return _parse_qa_response(text)


def _parse_qa_response(text: str) -> list[dict]:
    """Parse LLM response into QA pair list."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def _generate_rule_based(frames: list[dict], config: PipelineConfig) -> list[dict]:
    """
    Generate QA pairs using rule-based templates.
    Works without any API access — useful for testing and bootstrapping.
    """
    print(f"  Generating {config.questions_per_sample} QA pairs (rule-based)...")

    qa_pairs = []
    q_id = 1

    # Collect all objects across frames with their trajectories
    object_trajectories = _build_object_trajectories(frames)

    # --- Spatial recall questions (easy) ---
    for obj_id, trajectory in object_trajectories.items():
        if q_id > config.questions_per_sample:
            break
        first_appearance = trajectory[0]
        label = first_appearance["label"]
        frame = first_appearance["frame"]
        spatial = first_appearance.get("spatial_description", first_appearance.get("state", "in the scene"))

        qa_pairs.append({
            "question_id": f"q{q_id:03d}",
            "question": f"Where was the {label.replace('_', ' ')} at {frame['timestamp']}?",
            "answer": spatial,
            "answer_type": "fuzzy",
            "dimensions": ["spatial_recall"],
            "difficulty": "easy",
            "relevant_frames": [frame["frame_id"]],
        })
        q_id += 1

    # --- Temporal tracking questions (medium) ---
    for obj_id, trajectory in object_trajectories.items():
        if q_id > config.questions_per_sample:
            break
        if len(trajectory) < 2:
            continue

        first = trajectory[0]
        last = trajectory[-1]
        label = first["label"]

        if first.get("state") != last.get("state") or first.get("spatial_description") != last.get("spatial_description"):
            changes = []
            for t in trajectory:
                changes.append(f"{t.get('state', 'unknown')} ({t['frame']['frame_id']})")

            qa_pairs.append({
                "question_id": f"q{q_id:03d}",
                "question": f"Did the {label.replace('_', ' ')} move or change state during the sequence?",
                "answer": f"Yes, it went through these states: {', '.join(changes)}",
                "answer_type": "fuzzy",
                "dimensions": ["temporal_tracking", "spatial_recall"],
                "difficulty": "medium",
                "relevant_frames": [t["frame"]["frame_id"] for t in trajectory],
            })
            q_id += 1

    # --- Entity resolution / disappearance questions (medium) ---
    all_object_ids = set()
    last_frame_object_ids = set()
    for frame in frames:
        for obj in frame.get("objects", []):
            all_object_ids.add(obj.get("object_id", ""))
    if frames:
        for obj in frames[-1].get("objects", []):
            last_frame_object_ids.add(obj.get("object_id", ""))

    disappeared = all_object_ids - last_frame_object_ids
    for obj_id in disappeared:
        if q_id > config.questions_per_sample:
            break
        traj = object_trajectories.get(obj_id, [])
        if not traj:
            continue
        label = traj[0]["label"]
        last_seen = traj[-1]

        qa_pairs.append({
            "question_id": f"q{q_id:03d}",
            "question": f"When was the {label.replace('_', ' ')} last seen in the sequence?",
            "answer": f"Last seen at {last_seen['frame']['timestamp']} in frame {last_seen['frame']['frame_id']}",
            "answer_type": "fuzzy",
            "dimensions": ["temporal_tracking", "entity_resolution"],
            "difficulty": "medium",
            "relevant_frames": [t["frame"]["frame_id"] for t in traj],
        })
        q_id += 1

    # --- Contextual retrieval questions (hard) ---
    if len(frames) >= 2 and q_id <= config.questions_per_sample:
        first_frame = frames[0]
        last_frame = frames[-1]

        first_objects = {o.get("object_id"): o for o in first_frame.get("objects", [])}
        last_objects = {o.get("object_id"): o for o in last_frame.get("objects", [])}

        changes = []
        for oid, obj in first_objects.items():
            if oid not in last_objects:
                changes.append(f"The {obj['label'].replace('_', ' ')} is no longer visible")
            elif obj.get("state") != last_objects[oid].get("state"):
                changes.append(
                    f"The {obj['label'].replace('_', ' ')} changed from "
                    f"{obj.get('state', 'unknown')} to {last_objects[oid].get('state', 'unknown')}"
                )
        for oid, obj in last_objects.items():
            if oid not in first_objects:
                changes.append(f"The {obj['label'].replace('_', ' ')} appeared")

        if changes:
            qa_pairs.append({
                "question_id": f"q{q_id:03d}",
                "question": "What changed between the first and last frame in the sequence?",
                "answer": "; ".join(changes),
                "answer_type": "open_ended",
                "dimensions": ["temporal_tracking", "contextual_retrieval"],
                "difficulty": "hard",
                "relevant_frames": [first_frame["frame_id"], last_frame["frame_id"]],
            })
            q_id += 1

    print(f"  Generated {len(qa_pairs)} QA pairs")
    return qa_pairs[:config.questions_per_sample]


def _build_object_trajectories(frames: list[dict]) -> dict[str, list[dict]]:
    """
    Build a trajectory for each object: ordered list of appearances across frames.
    """
    trajectories: dict[str, list[dict]] = {}

    for frame in frames:
        for obj in frame.get("objects", []):
            obj_id = obj.get("object_id", "")
            if not obj_id:
                continue
            if obj_id not in trajectories:
                trajectories[obj_id] = []
            trajectories[obj_id].append({
                "frame": frame,
                "label": obj.get("label", ""),
                "state": obj.get("state", ""),
                "spatial_description": obj.get("spatial_description", ""),
                "bounding_box": obj.get("bounding_box", {}),
            })

    return trajectories
