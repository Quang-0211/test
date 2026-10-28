# PRDetect analysis

## Reproducibility record

- Repository: `thulx18/PRDetect`
- Checked-out commit: `83ebc6e2cd543477f83e405d702ac837e664fc9e`
- The checkout contains 7 Python files and 19 notebooks. It contains source JSON/JSONL
  and perturbation directories, but no `.pkl` graph files, `roberta-base/` directory, or
  trained checkpoints. Numerical results below are therefore not re-run results.
- `requirements.txt` pins spaCy 3.7.4, Transformers 4.40.0, PyG 2.5.2 and a Linux
  Python 3.8 CUDA 11.3 PyTorch wheel. The available host interpreter is Python 3.12
  without PyTorch, so the environment and model tests remain pending.

## Data flow verified from source

`original_text/*.json` -> `building_graph.read_json` -> spaCy tokens and RoBERTa
hidden states -> `all_token_embeddings`, `all_edge_index`, `y` -> pickle -> `gcn.py`
training/validation -> checkpoint -> `test.py` metrics or `detect.py` prediction.

Each graph stores a list of `x` tensors with shape `[N, 768]`, normally float tensors;
`edge_index` has shape `[2, E]` and `torch.long` dtype; and `y` is a float tensor.
The original builder creates one edge per spaCy token as `[word.i, word.head.i]`
(dependent to head), including the root self-loop. The implementation now also stores `all_edge_type`, shape `[E]`, `torch.long`, with IDs
derived from the installed spaCy parser label list. R-GCN preparation removes root
self-loops and optionally appends reverse edges.

## Model and training

The original `model/GCN2.py` and the notebook/script define GCNConv 768->256,
ReLU/dropout(0.5), GCNConv 256->64, ReLU/dropout, node Linear 64->1, node mean,
and sigmoid. Training uses Adam at 1e-4, BCELoss, one graph per step, and up to 40
epochs. The old script hard-coded paths, seed 2024, and dataset `hc3`; its stopping
counter was reset inside the epoch loop.

The new CLI defaults preserve the GCN dimensions and optimizer while adding
`--model`, `--seed`, `--dataset`, `--num_bases`, `--add_reverse`, `--use_labels`,
`--graph_dir`, and `--out_dir`. GCN checkpoints retain the `*_gcn_model_*` name;
R-GCN uses `*_rgcn_model_*`.

## Verified discrepancies and fixes

| Finding | Status |
|---|---|
| `time` and `device` missing in `building_graph.py` | Confirmed; fixed minimally |
| `/root/autodl-tmp` hard-coded | Confirmed; replaced by CLI-relative directories |
| `test.py` parser destination mismatch | Confirmed; fixed with `dest` names |
| `--seed` was a string | Confirmed; now `int` |
| `model/GCN2.py` existed despite the old `detect.py` assumptions | Confirmed; imports are now centralized |
| `detect.py` passed a JSON string rather than a list | Confirmed; fixed |
| early-stop counter reset | Confirmed in old script; new loop selects best validation checkpoint |
| license page visibility | Not independently verified; source checkout has no root LICENSE |

Perturbation notebooks and JSON files are present, but no perturbation experiment was
executed. Rebuilding each perturbed set is required because spaCy may change both
edges and dependency labels.
