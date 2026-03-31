---
name: Dataset Contribution
about: Submit annotated data for the benchmark
title: "[Data] "
labels: dataset-contribution
assignees: ''
---

## Source Dataset
<!-- Which dataset did you annotate? (e.g. Ego4D, ScanNet, custom camera footage) -->

## Number of Samples
<!-- How many evaluation samples are you contributing? -->

## Evaluation Dimensions Covered
- [ ] Spatial Recall
- [ ] Temporal Tracking
- [ ] Entity Resolution
- [ ] Cross-modal Fusion
- [ ] Hot/Cold Memory Tiering
- [ ] Contextual Retrieval

## Schema Validation
<!-- Did your samples pass validation? -->
```bash
python evaluation/validate_schema.py --file your_file.json
```
- [ ] Yes, all samples pass validation

## Notes
<!-- Any additional context about the data or annotations -->
