# R-GCN experiment report

## Status

Implementation is complete, but experiments are **not run**. The checkout has no
precomputed graph pickles or model checkpoints, and the current host has Python 3.12
without PyTorch. It would be invalid to report accuracy, F1, AUC, timing, or statistical
significance.

## Required execution

1. Create the pinned Python 3.8 environment, install `requirements.txt`, download
   `en_core_web_sm`, and place `roberta-base` in `./roberta-base/`.
2. Build legacy graphs with `python building_graph.py`.
3. Add labels with `python add_edge_type.py --files hc3_train hc3_val hc3_test`.
4. Train each configuration with `python gcn.py --model ...` and evaluate using
   `python test.py --file hc3_test --save`.
5. Add perturbation and cross-dataset runs only after rebuilding their graphs.

The requested five-seed tables, perturbation label-change rates, parameter/timing
comparisons, and paired tests remain pending and must be populated from actual runs.
