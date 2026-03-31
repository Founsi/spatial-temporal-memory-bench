#!/usr/bin/env python3
"""
Spatial-Temporal Memory Bench — Curation Pipeline

End-to-end pipeline for curating evaluation samples from raw video or image data.

Usage:
    # From video (requires opencv-python):
    python -m pipeline.run_pipeline --input video.mp4 --source video --env kitchen

    # From image directory:
    python -m pipeline.run_pipeline --input ./frames/ --source image_dir --env office

    # With vision API (OpenAI):
    python -m pipeline.run_pipeline --input video.mp4 --source video --env kitchen \\
        --vision-provider openai --vision-model gpt-4o

    # With vision API (Anthropic):
    python -m pipeline.run_pipeline --input video.mp4 --source video --env kitchen \\
        --vision-provider anthropic --vision-model claude-sonnet-4-20250514

    # Local/offline mode (no API calls, uses placeholders):
    python -m pipeline.run_pipeline --input ./frames/ --source image_dir --env kitchen \\
        --vision-provider local --qa-provider local

    # Pipeline + launch review UI automatically:
    python -m pipeline.run_pipeline --input ./frames/ --source image_dir --env kitchen --review

    # Review existing curated samples (skip pipeline):
    python -m pipeline.run_pipeline --review-only --output-dir data/curated
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig
from pipeline.extract import extract_frames
from pipeline.analyze import analyze_frames
from pipeline.track import track_objects
from pipeline.generate_qa import generate_qa_pairs
from pipeline.assemble import assemble_sample, save_sample


def run_pipeline(
    config: PipelineConfig,
    skip_extract: bool = False,
    launch_review: bool = False,
    review_port: int = 8765,
) -> Path:
    """
    Run the full curation pipeline.

    Stages:
        1. Extract keyframes from video/images
        2. Analyze each frame with a vision model
        3. Track objects across frames with persistent IDs
        4. Generate QA pairs for memory evaluation
        5. Assemble and save schema-valid annotation JSON
        6. (Optional) Launch human review UI

    Returns the path to the saved annotation file.
    """
    total_stages = 6 if launch_review else 5
    print("=" * 60)
    print("Spatial-Temporal Memory Bench — Curation Pipeline")
    print("=" * 60)

    # Stage 1: Extract
    print(f"\n[1/{total_stages}] Extracting frames...")
    if skip_extract:
        print("  Skipped (--skip-extract)")
        frames = extract_frames(config)  # image_dir mode just loads existing images
    else:
        frames = extract_frames(config)

    if not frames:
        print("ERROR: No frames extracted. Check your input path.")
        sys.exit(1)

    # Stage 2: Analyze
    print(f"\n[2/{total_stages}] Analyzing scenes...")
    frames = analyze_frames(frames, config)

    # Stage 3: Track
    print(f"\n[3/{total_stages}] Tracking objects...")
    frames = track_objects(frames, config)

    # Stage 4: Generate QA
    print(f"\n[4/{total_stages}] Generating QA pairs...")
    qa_pairs = generate_qa_pairs(frames, config)

    # Stage 5: Assemble
    print(f"\n[5/{total_stages}] Assembling annotation...")
    sample = assemble_sample(frames, qa_pairs, config)
    output_path = save_sample(sample, config)

    print("\n" + "=" * 60)
    print(f"Pipeline complete!")
    print(f"  Output: {output_path}")
    print(f"  Frames: {len(frames)}")
    print(f"  Objects tracked: {len({o['object_id'] for f in frames for o in f.get('objects', [])})}")
    print(f"  QA pairs: {len(qa_pairs)}")

    # Stage 6: Human review
    if launch_review:
        print(f"\n[6/{total_stages}] Launching human review UI...")
        print("=" * 60)
        _launch_review(config.output_dir, review_port)
    else:
        print(f"\n  To review: python -m pipeline.review.server --samples {config.output_dir}")
        print("=" * 60)

    return output_path


def _launch_review(samples_path: Path, port: int):
    """Launch the review server."""
    from pipeline.review.server import ReviewServer, create_handler
    from http.server import HTTPServer

    project_root = Path(__file__).parent.parent
    review_server = ReviewServer(samples_path, project_root)
    handler_class = create_handler(review_server)

    httpd = HTTPServer(("127.0.0.1", port), handler_class)
    print(f"\n  Review UI running at: http://127.0.0.1:{port}")
    print(f"  Reviewing {len(review_server.samples)} sample(s)")
    print(f"  Press Ctrl+C to stop\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nReview complete. Shutting down.")
        httpd.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Curate spatial-temporal memory evaluation samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    parser.add_argument("--input", default=None, help="Path to video file or image directory (required unless --review-only)")
    parser.add_argument("--source", default="video", choices=["video", "image_dir"],
                        help="Source type (default: video)")
    parser.add_argument("--env", default="unknown", help="Environment name (e.g., kitchen, office)")

    # Frame extraction
    parser.add_argument("--max-frames", type=int, default=20, help="Maximum frames to extract")
    parser.add_argument("--frame-interval", type=float, default=30.0,
                        help="Seconds between extracted frames (default: 30)")
    parser.add_argument("--scene-change", action="store_true",
                        help="Use scene-change detection instead of fixed interval")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction (use with image_dir source)")

    # Vision analysis
    parser.add_argument("--vision-provider", default="local",
                        choices=["openai", "anthropic", "local"],
                        help="Vision model provider (default: local)")
    parser.add_argument("--vision-model", default=None,
                        help="Vision model name (defaults based on provider)")

    # QA generation
    parser.add_argument("--qa-provider", default="local",
                        choices=["openai", "anthropic", "local"],
                        help="LLM for QA generation (default: local)")
    parser.add_argument("--qa-model", default=None,
                        help="QA model name (defaults based on provider)")
    parser.add_argument("--num-questions", type=int, default=8,
                        help="Number of QA pairs to generate (default: 8)")

    # Output
    parser.add_argument("--output-dir", default="data/curated",
                        help="Output directory (default: data/curated)")
    parser.add_argument("--annotator", default="pipeline-auto",
                        help="Annotator name for metadata")

    # Human review
    parser.add_argument("--review", action="store_true",
                        help="Launch review UI after pipeline completes")
    parser.add_argument("--review-only", action="store_true",
                        help="Skip pipeline, just launch review UI on existing samples")
    parser.add_argument("--review-port", type=int, default=8765,
                        help="Port for review UI (default: 8765)")

    args = parser.parse_args()

    # Review-only mode: skip pipeline, just launch review server
    if args.review_only:
        print("Launching review UI (pipeline skipped)...")
        _launch_review(Path(args.output_dir), args.review_port)
        return

    if not args.input:
        parser.error("--input is required unless using --review-only")

    # Build config
    vision_model_defaults = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
        "local": "local",
    }
    qa_model_defaults = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
        "local": "local",
    }

    config = PipelineConfig(
        input_path=Path(args.input),
        source_type=args.source,
        environment=args.env,
        max_frames=args.max_frames,
        frame_interval_sec=args.frame_interval,
        use_scene_change=args.scene_change,
        vision_provider=args.vision_provider,
        vision_model=args.vision_model or vision_model_defaults[args.vision_provider],
        qa_provider=args.qa_provider,
        qa_model=args.qa_model or qa_model_defaults[args.qa_provider],
        questions_per_sample=args.num_questions,
        output_dir=Path(args.output_dir),
        image_output_dir=Path(args.output_dir) / "images",
        annotator=args.annotator,
    )

    run_pipeline(
        config,
        skip_extract=args.skip_extract,
        launch_review=args.review,
        review_port=args.review_port,
    )


if __name__ == "__main__":
    main()
