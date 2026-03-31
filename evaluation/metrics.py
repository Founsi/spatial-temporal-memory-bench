"""
Spatial-Temporal Memory Bench — Evaluation Metrics

Scoring functions for memory retrieval evaluation.
This is where we need the most community input — temporal visual
memory metrics don't exist yet in the literature.
"""

from difflib import SequenceMatcher
from typing import Any


def score_exact_match(predicted: str, ground_truth: str) -> float:
    """Exact string match (case-insensitive). Returns 0 or 1."""
    return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0


def score_fuzzy_match(predicted: str, ground_truth: str) -> float:
    """
    Fuzzy string similarity using SequenceMatcher.
    Returns a score between 0 and 1.

    This is a placeholder — we should replace this with a more
    sophisticated semantic similarity metric (e.g. embedding cosine
    similarity or LLM-as-judge).
    """
    if not predicted or not ground_truth:
        return 0.0
    return SequenceMatcher(
        None,
        predicted.strip().lower(),
        ground_truth.strip().lower(),
    ).ratio()


def compute_dimension_scores(results: list[dict[str, Any]]) -> dict[str, float]:
    """
    Compute average scores per evaluation dimension.

    Each question can test multiple dimensions, so a single question's
    score contributes to all its tagged dimensions.
    """
    dimension_totals: dict[str, list[float]] = {}

    for r in results:
        for dim in r["dimensions"]:
            if dim not in dimension_totals:
                dimension_totals[dim] = []
            dimension_totals[dim].append(r["score"])

    return {
        dim: round(sum(scores) / len(scores), 4)
        for dim, scores in dimension_totals.items()
        if scores
    }


# ---------------------------------------------------------------------------
# TODO: Community contributions needed for these metrics
# ---------------------------------------------------------------------------
#
# 1. Temporal Accuracy Score
#    - How accurately does the system track *when* things changed?
#    - Should penalize temporal misalignment (e.g. "moved at 3pm" vs "moved at 5pm")
#
# 2. Spatial Precision Score
#    - How accurately does the system recall *where* objects are?
#    - Could use IoU of bounding boxes or distance-based scoring
#
# 3. Entity Resolution Score
#    - Can the system correctly identify the same object across frames?
#    - Precision/recall on object identity matching
#
# 4. Temporal Decay Score
#    - How does retrieval accuracy degrade with time distance?
#    - Compare hot memory (recent) vs cold memory (older) performance
#
# 5. Semantic Similarity Score
#    - Replace fuzzy string matching with embedding-based scoring
#    - Or use an LLM-as-judge approach for open-ended answers
#
# If you'd like to implement any of these, open an issue or submit a PR!
