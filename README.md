# Spatial-Temporal Memory Bench

**The first open-source evaluation dataset for visual memory systems operating in changing spatial and temporal environments.**

Most existing memory benchmarks (LongMemEval, MEMTRACK, MemoryAgentBench, Letta Leaderboard) only test text-based recall. As multimodal AI matures, long-term visual memory will be the key to high-performing agentic systems. Yet there is no standard way to evaluate whether a memory system can track objects, reason about spatial changes, or recall temporal context from visual input.

This project aims to change that.

## Why This Matters

Agentic systems are increasingly operating in physical and visual environments — robotics, smart homes, autonomous navigation, personal AI assistants. These systems need memory that goes beyond text. They need to answer questions like:

- "Where was my phone last seen?"
- "What changed in this room since Tuesday?"
- "Has this object moved in the last 10 days?"

No benchmark exists to evaluate these capabilities. We're building one.

## Evaluation Tracks

### Track 1: Visual Object Tracking from Images/Video

Evaluate whether memory systems can track objects' spatial and temporal parameters from sequential images or video frames. Think of a home camera (e.g. Wyze) capturing snapshots at intervals — can a memory system ingest these frames and answer retrieval questions about object state, location, and change over time?

**What we test:**
- Object location recall ("Where is the backpack?")
- State change detection ("What moved since this morning?")
- Temporal reasoning ("When was the last time the door was open?")
- Entity resolution across frames (same object, different angles/lighting)

### Track 2: World Model and Robotics Dataset Adaptation

Leverage existing world model and robotics datasets that already contain rich temporal, multimodal sensor data (visual, audio, spatial). We transform these by adding QA pairs specifically designed to evaluate memory retrieval and reasoning.

**What we test:**
- Cross-modal consolidation (visual + spatial + temporal)
- Hot memory recall (recent events, fast retrieval)
- Cold memory recall (older events, requires deeper search)
- Context-based retrieval (gist/meaning of scenes, not just structured attributes)

## Proposed Evaluation Dimensions

| Dimension | Description | Example Question |
|-----------|-------------|------------------|
| Spatial Recall | Remember where objects are | "Where was the mug last seen?" |
| Temporal Tracking | Track state changes over time | "What changed in the kitchen since 3pm?" |
| Entity Resolution | Identify same object across views | "Is this the same package from yesterday?" |
| Cross-modal Fusion | Combine visual + other modalities | "The sound came from the direction of..." |
| Hot/Cold Tiering | Test both recent and older memories | Recent vs. 10-day-old retrieval |
| Contextual Retrieval | Recall scene meaning, not just facts | "What was happening when the lights went out?" |

## Data Format

See [`data/schemas/`](data/schemas/) for the proposed annotation schema. Each evaluation sample consists of:

1. **Visual sequence**: Ordered set of images or video frames with timestamps
2. **Annotations**: Object locations, states, and changes across the sequence
3. **QA pairs**: Questions and ground-truth answers for evaluation
4. **Metadata**: Source dataset, difficulty level, evaluation dimensions tested

## Getting Started

```bash
# Clone the repo
git clone https://github.com/YOUR_ORG/spatial-temporal-memory-bench.git
cd spatial-temporal-memory-bench

# Install evaluation dependencies
pip install -r requirements.txt

# Run evaluation on a sample
python evaluation/run_eval.py --data data/examples/sample_sequence.json --model YOUR_MODEL
```

## Contributing

We are actively looking for contributors. This is a community-driven effort uniting engineers across companies working on the memory problem.

**Ways to contribute:**
- Annotate existing visual datasets with memory QA pairs
- Propose new evaluation scenarios and metrics
- Submit evaluation scripts for additional memory systems
- Share relevant world model or robotics datasets

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Project Context

This benchmark is part of the [Agent Memory Benchmark](https://agentmemorybenchmark.ai) initiative, bringing together researchers and engineers from across the AI memory ecosystem to build shared evaluation infrastructure.

### Related Benchmarks
- [LongMemEval](https://arxiv.org/abs/2410.10813) — Long-term memory evaluation (text-only)
- [MEMTRACK](https://arxiv.org/abs/2410.02757) — Memory tracking benchmark by Patronus AI (text-only)
- [MemoryAgentBench](https://arxiv.org/abs/2501.13476) — Agent memory benchmark (ICLR 2026)
- [Letta Leaderboard](https://letta.com/leaderboard) — MemGPT-based memory evaluation

### Key References
- "Generative Agents: Interactive Simulacra of Human Behavior" — [arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
