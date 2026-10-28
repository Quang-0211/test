# Implementation status

Evidence was checked on 2026-10-28.

| Stage | Status | Evidence |
|---|---|---|
| 0. Preparation | XONG | Both repositories are cloned; commits and Kaggle environment notes are in [ANALYSIS_PRDetect.md](./ANALYSIS_PRDetect.md). |
| 1. PRDetect analysis | XONG | [ANALYSIS_PRDetect.md](./ANALYSIS_PRDetect.md) covers source scripts, notebooks, data flow, discrepancies, and missing `.pkl` files. |
| 2. Torch-RGCN analysis | XONG | [ANALYSIS_TorchRGCN.md](./ANALYSIS_TorchRGCN.md) records the PyG `RGCNConv` decision and Torch-RGCN exclusions. |
| 3. Design | XONG | [DESIGN.md](./DESIGN.md) records relation mapping, self-loops, reverse edges, model, preprocessing, and CLI decisions. |
| 4. Implementation | XONG | [model/RGCN2.py](./model/RGCN2.py), [add_edge_type.py](./add_edge_type.py), and the train/test/detect scripts are present; [CHANGELOG.md](./CHANGELOG.md) records changes. |
| 5. Tests | DANG DỞ | Syntax and static checks ran; CPU PyTorch tests are not runnable because this host has no PyTorch. See [TESTS.md](./TESTS.md). |
| 6. Kaggle preparation | XONG | The `kaggle/` runner, configuration, notebook, aggregation, stability, and runtime guidance files are present. |
| 7. Handover docs | XONG | [REPORT_TEMPLATE.md](./REPORT_TEMPLATE.md) and [README_RGCN.md](./README_RGCN.md) are present. |

No training, real-data graph building, RoBERTa download, or experiment result was
performed locally. Those remain Kaggle work.
