# R-GCN experiment report

All cells below are intentionally `chưa chạy, sẽ chạy trên Kaggle`.

## Results

| Config | Dataset | Evaluation | Seeds | Accuracy mean ± std | F1 mean ± std | AUC mean ± std |
|---|---|---|---|---|---|---|
| A0 | HC3 | Original | 2021-2025 | chưa chạy | chưa chạy | chưa chạy |
| B0 | HC3 | Original | 2021-2025 | chưa chạy | chưa chạy | chưa chạy |
| B2 | HC3 | Replace/Delete 25% | 2021-2025 | chưa chạy | chưa chạy | chưa chạy |

## Cost

| Config | Parameters | Preprocessing seconds | Training seconds | Test seconds |
|---|---:|---:|---:|---:|
| A0 | chưa chạy | chưa chạy | chưa chạy | chưa chạy |
| B2 | chưa chạy | chưa chạy | chưa chạy | chưa chạy |

## Dependency-label stability

The planned measurement matches tokens by aligned token position where both parsed
documents have a token at that position, and counts changed dependency labels over
those aligned tokens. Insert/delete operations can shift positions; unmatched tokens
are excluded and reported separately. Values: chưa chạy, sẽ chạy trên Kaggle.

## Open verification items

- PyG version and `RGCNConv` signature on the Kaggle image.
- GCN A0 reproduction on the same Kaggle runtime.
- All five seeds and all requested evaluation sets.
- Paired significance test and parameter-matched C1/D ablations.
