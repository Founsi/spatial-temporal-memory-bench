"""
Spatial-Temporal Memory Bench — Curation Pipeline

End-to-end pipeline for curating evaluation samples from raw video/image data.

Stages:
    1. extract   — Extract keyframes from video or load image sequences
    2. analyze   — Detect objects and describe scenes using vision models
    3. track     — Assign persistent object IDs across frames
    4. generate  — Auto-generate QA pairs for memory evaluation
    5. assemble  — Combine into schema-valid annotation JSON
"""
