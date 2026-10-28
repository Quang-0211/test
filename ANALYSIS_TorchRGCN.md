# Torch-RGCN analysis

- Repository: `thiviyanT/torch-rgcn`
- Checked-out commit: `267faffd09a441d902c483a8c130410c72910e90`
- License: MIT (verified in `LICENSE`).

The repository is a knowledge-graph implementation. Its entry points are
`experiments/classify_nodes.py` and `experiments/predict_links.py`; YAML files under
`configs/` configure node classification and link prediction experiments.
`RelationalGraphConvolutionNC` in `torch_rgcn/layers.py` consumes triples and can
construct dense node features or featureless one-hot-like inputs. It implements
relation weights, basis/block decomposition, optional inverse/self-loop relations,
and degree normalization. `NodeClassifier` adds inverse and self-loop triples.
`LinkPredictor` additionally uses node embeddings and a DistMult decoder.

Those triple batching, link sampling, DistMult, featureless node initialization,
and graph-specific pruning paths are not used in PRDetect. PRDetect has many small
graphs, dense RoBERTa features `[N, 768]`, and one document-level label.

## Decision

The integration uses PyTorch Geometric `RGCNConv` in `model/RGCN2.py`, not a copied
Torch-RGCN layer. This is the least invasive choice because PyG directly accepts
`x`, `edge_index`, and `edge_type` for each small graph and provides the requested
`num_bases`/`num_blocks` controls. Compatibility with the pinned Python 3.8/PyTorch
1.11 environment and numerical equivalence tests are **not run** on this host because
PyTorch is not installed. A shape/gradient test remains a required environment step.
