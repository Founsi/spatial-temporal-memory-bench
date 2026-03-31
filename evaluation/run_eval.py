"""
Spatial-Temporal Memory Bench — Evaluation Runner

Runs a memory system against the benchmark and reports scores
across evaluation dimensions.

Usage:
    python evaluation/run_eval.py --data data/examples/sample_sequence.json --model dummy
"""

import argparse
import json
from pathlib import Path
from typing import Any

from metrics import (
    score_exact_match,
    score_fuzzy_match,
    compute_dimension_scores,
)
from validate_schema import validate_sample


def load_sample(path: str) -> dict:
    """Load and validate a single evaluation sample."""
    with open(path) as f:
        sample = json.load(f)
    errors = validate_sample(sample)
    if errors:
        raise ValueError(f"Schema validation failed: {errors}")
    return sample


def run_model(model_name: str, sample: dict) -> list[dict]:
    """
    Run a memory model against the QA pairs in a sample.

    Returns a list of dicts with 'question_id' and 'predicted_answer'.

    To add your own model, create an adapter in evaluation/adapters/
    and register it here.
    """
    adapters = {
        "dummy": _dummy_adapter,
        # Add your adapters here:
        # "mem0": adapters.mem0_adapter.run,
        # "letta": adapters.letta_adapter.run,
        # "zep": adapters.zep_adapter.run,
    }

    if model_name not in adapters:
        available = ", ".join(adapters.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available}"
        )

    return adapters[model_name](sample)


def _dummy_adapter(sample: dict) -> list[dict]:
    """Dummy adapter that returns empty answers. For testing the harness."""
    return [
        {"question_id": qa["question_id"], "predicted_answer": ""}
        for qa in sample["qa_pairs"]
    ]


def evaluate(sample: dict, predictions: list[dict]) -> dict[str, Any]:
    """Score predictions against ground truth."""
    pred_map = {p["question_id"]: p["predicted_answer"] for p in predictions}

    results = []
    for qa in sample["qa_pairs"]:
        qid = qa["question_id"]
        predicted = pred_map.get(qid, "")
        ground_truth = qa["answer"]
        answer_type = qa.get("answer_type", "fuzzy")

        if answer_type == "exact":
            score = score_exact_match(predicted, ground_truth)
        else:
            score = score_fuzzy_match(predicted, ground_truth)

        results.append({
            "question_id": qid,
            "question": qa["question"],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "score": score,
            "dimensions": qa["dimensions"],
            "difficulty": qa.get("difficulty", "medium"),
        })

    dimension_scores = compute_dimension_scores(results)

    overall = (
        sum(r["score"] for r in results) / len(results)
        if results
        else 0.0
    )

    return {
        "overall_score": round(overall, 4),
        "dimension_scores": dimension_scores,
        "num_questions": len(results),
        "detailed_results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run spatial-temporal memory evaluation"
    )
    parser.add_argument(
        "--data", required=True, help="Path to evaluation sample JSON"
    )
    parser.add_argument(
        "--model", default="dummy", help="Model adapter to use"
    )
    parser.add_argument(
        "--output", default=None, help="Path to save results JSON"
    )
    args = parser.parse_args()

    print(f"Loading sample from {args.data}...")
    sample = load_sample(args.data)

    print(f"Running model '{args.model}'...")
    predictions = run_model(args.model, sample)

    print("Evaluating...")
    results = evaluate(sample, predictions)

    print(f"\n{'='*50}")
    print(f"Overall Score: {results['overall_score']}")
    print(f"Questions Evaluated: {results['num_questions']}")
    print(f"\nScores by Dimension:")
    for dim, score in results["dimension_scores"].items():
        print(f"  {dim}: {score}")
    print(f"{'='*50}\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
