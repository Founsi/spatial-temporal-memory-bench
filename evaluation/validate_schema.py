"""
Schema validation for evaluation samples.

Usage:
    python evaluation/validate_schema.py --file path/to/annotation.json
"""

import argparse
import json
from pathlib import Path


SCHEMA_PATH = Path(__file__).parent.parent / "data" / "schemas" / "annotation_schema.json"

REQUIRED_FIELDS = ["id", "version", "visual_sequence", "qa_pairs", "metadata"]
VALID_DIMENSIONS = {
    "spatial_recall",
    "temporal_tracking",
    "entity_resolution",
    "cross_modal_fusion",
    "hot_memory",
    "cold_memory",
    "contextual_retrieval",
}
VALID_ANSWER_TYPES = {"exact", "fuzzy", "multiple_choice", "open_ended"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def validate_sample(sample: dict) -> list[str]:
    """
    Validate a sample against the annotation schema.
    Returns a list of error messages (empty if valid).

    This is a lightweight validator that doesn't require jsonschema.
    For full JSON Schema validation, install jsonschema and use
    validate_sample_strict() instead.
    """
    errors = []

    # Check required top-level fields
    for field in REQUIRED_FIELDS:
        if field not in sample:
            errors.append(f"Missing required field: {field}")

    if errors:
        return errors

    # Validate visual_sequence
    vs = sample["visual_sequence"]
    if "frames" not in vs:
        errors.append("visual_sequence must contain 'frames'")
    elif not isinstance(vs["frames"], list):
        errors.append("visual_sequence.frames must be an array")
    else:
        for i, frame in enumerate(vs["frames"]):
            if "frame_id" not in frame:
                errors.append(f"Frame {i} missing 'frame_id'")
            if "timestamp" not in frame:
                errors.append(f"Frame {i} missing 'timestamp'")
            if "image_path" not in frame:
                errors.append(f"Frame {i} missing 'image_path'")

    # Validate qa_pairs
    qa = sample["qa_pairs"]
    if not isinstance(qa, list):
        errors.append("qa_pairs must be an array")
    else:
        for i, pair in enumerate(qa):
            if "question_id" not in pair:
                errors.append(f"QA pair {i} missing 'question_id'")
            if "question" not in pair:
                errors.append(f"QA pair {i} missing 'question'")
            if "answer" not in pair:
                errors.append(f"QA pair {i} missing 'answer'")
            if "dimensions" not in pair:
                errors.append(f"QA pair {i} missing 'dimensions'")
            elif not isinstance(pair["dimensions"], list):
                errors.append(f"QA pair {i} 'dimensions' must be an array")
            else:
                for dim in pair["dimensions"]:
                    if dim not in VALID_DIMENSIONS:
                        errors.append(
                            f"QA pair {i} has invalid dimension '{dim}'. "
                            f"Valid: {VALID_DIMENSIONS}"
                        )

            if "answer_type" in pair and pair["answer_type"] not in VALID_ANSWER_TYPES:
                errors.append(
                    f"QA pair {i} has invalid answer_type '{pair['answer_type']}'"
                )
            if "difficulty" in pair and pair["difficulty"] not in VALID_DIFFICULTIES:
                errors.append(
                    f"QA pair {i} has invalid difficulty '{pair['difficulty']}'"
                )

    # Validate metadata
    meta = sample["metadata"]
    if "source_dataset" not in meta:
        errors.append("metadata missing 'source_dataset'")
    if "annotator" not in meta:
        errors.append("metadata missing 'annotator'")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate annotation schema")
    parser.add_argument("--file", required=True, help="Path to annotation JSON")
    args = parser.parse_args()

    with open(args.file) as f:
        sample = json.load(f)

    errors = validate_sample(sample)
    if errors:
        print(f"Validation FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        exit(1)
    else:
        print("Validation PASSED")


if __name__ == "__main__":
    main()
