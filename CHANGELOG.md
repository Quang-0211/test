# Changelog

## R-GCN integration

- `model/RGCN2.py`: added the relation-aware two-layer classifier with the original
  768->256->64 architecture, dropout, mean pooling, and sigmoid.
- `model/factory.py`: centralized GCN/R-GCN construction so train, test, and detect
  cannot silently diverge.
- `building_graph.py`: imported the missing `time`/device handling, added stable
  bounded dependency labels, saved `all_edge_type`, and replaced the hard-coded output
  path with `graph_data`.
- `add_edge_type.py`: added a no-RoBERTa migration utility that validates old edges and
  token counts before adding relation labels.
- `gcn.py`: added model/configuration flags, portable graph paths, relation/reverse-edge
  validation, and separate R-GCN checkpoints while preserving GCN defaults.
- `test.py`: corrected CLI destinations and integer seed handling, added both model
  types, and recorded the requested result metadata.
- `detect.py`: corrected the graph-builder call and added R-GCN relation preparation.
- `building_graph.py`: relation IDs now come from the installed spaCy parser labels
  rather than a lossy hash, and the relation vocabulary size is derived dynamically.
- `test.py` and `kaggle/run_matrix.py`: added machine-readable per-run CSV output for
  resumable Kaggle execution.

No training or benchmark result is claimed: the host lacks the pinned ML environment
and the checkout contains no graph pickle/checkpoint artifacts.
