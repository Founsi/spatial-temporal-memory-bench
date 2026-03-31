# Contributing to Spatial-Temporal Memory Bench

Thanks for your interest in contributing. This is a community-driven project and we welcome help from anyone working on the memory problem in AI.

## How You Can Contribute

### 1. Annotate Existing Datasets

The fastest way to grow the benchmark is to add memory evaluation QA pairs to existing visual datasets. Pick a world model or robotics dataset, create annotation entries following our [schema](data/schemas/annotation_schema.json), and submit a PR.

**Good source datasets to start with:**
- Ego4D (egocentric video with temporal annotations)
- RoboTurk / RoboNet (robotic manipulation sequences)
- ScanNet (indoor 3D scene understanding)
- Habitat / AI2-THOR (simulated indoor environments)
- Any home camera footage you have rights to share

### 2. Propose Evaluation Scenarios

Open an issue with tag `scenario-proposal` describing:
- The scenario (e.g. "Track kitchen items across a day")
- What memory dimensions it tests (spatial recall, temporal tracking, etc.)
- Why existing benchmarks don't cover it
- Suggested data sources

### 3. Add Evaluation Metrics

We need new metrics specifically designed for temporal visual memory. If you have ideas for how to score a system's ability to track spatial changes over time, we want to hear them. See the `evaluation/` directory for the current metrics and submit a PR with your additions.

### 4. Submit Evaluation Scripts

If you've built or work with a memory system and want to evaluate it against this benchmark, add an adapter in `evaluation/adapters/` that connects your system to our evaluation harness.

### 5. Improve Documentation

Better docs, clearer schemas, more examples — all welcome.

## Submission Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-contribution`)
3. Make your changes
4. Ensure your annotations validate against the schema:
   ```bash
   python evaluation/validate_schema.py --file your_annotation.json
   ```
5. Submit a pull request with a clear description of what you're adding and why

## Annotation Guidelines

When creating QA pairs for the benchmark:

- **Be specific**: "Where is the red mug?" is better than "Where is the thing?"
- **Test real memory**: Questions should require information from previous frames, not just the current one
- **Include temporal reasoning**: At least some questions should involve time ("since when", "how long", "what changed")
- **Vary difficulty**: Include easy (single-frame recall), medium (multi-frame tracking), and hard (long-range temporal reasoning) questions
- **Use persistent object IDs**: The same object across frames should share an `object_id` so entity resolution can be tested

## Code Style

- Python code should follow PEP 8
- Include docstrings for all public functions
- Add type hints where practical
- JSON annotations should validate against the schema before submission

## Questions?

Open an issue or reach out to the maintainers. We're happy to help you find a good first contribution.
