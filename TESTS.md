# Validation record

## Ran locally

- `python -m py_compile building_graph.py add_edge_type.py gcn.py test.py detect.py model/*.py`
  passed.
- `git diff --check` passed.
- Static review confirmed that model construction is centralized, R-GCN data carries
  `edge_type`, paths are parameterized, and GCN/R-GCN checkpoints are distinct.
- Repository status and commit hashes were inspected for both repositories.

## Not run locally

- RGCN2 forward/gradient tests: not run; PyTorch and PyG are not installed.
- Five-sentence spaCy graph test: not run; spaCy and `en_core_web_sm` are not installed.
- Real-data graph building, RoBERTa encoding, training, and experiments: deliberately
  not run because the machine has no GPU and the task reserves experiments for Kaggle.

## Kaggle smoke command

```text
python kaggle/run_matrix.py --only A0,B0,B2 --seeds 2024
```

The runner writes one CSV row immediately after each completed train/test pair.
