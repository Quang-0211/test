# R-GCN integration design

1. Preserve the existing GCN path and checkpoint names. Select the new path only with
   `--model rgcn`.
2. Derive `LABELS` from the installed spaCy parser and map each label to its exact
   parser index. Reserve the second half of the relation vocabulary for reverse
   edges, so `num_relations=2 * len(LABELS)`.
3. Keep the original root self-loop in legacy graph data. The R-GCN data adapter removes
   self-loops before convolution because `RGCNConv` already has a root transform.
4. `RGCN2` has exactly the GCN2 dimensions and dropout, with two `RGCNConv` layers and
   default basis count 8. Mean pooling and sigmoid are unchanged.
5. `add_edge_type.py` reparses original JSON beside existing pickles, validates sample
   count, token count, and exact old `edge_index`, then writes `all_edge_type`. It fails
   loudly on mismatch and never silently rewrites embeddings.
6. `--add_reverse` controls whether the adapter adds reverse edges; this supports B1
   (forward only) and B2 (forward plus reverse) without changing stored embeddings.
7. Results are append-only in `test_result.txt`; model, bases, seed, direction/label
   flags, metrics, parameter count, and elapsed time are recorded.

Unresolved scientific choices (grouped dependency labels, block ablations, paired
statistics, and perturbation matching) are experiment/report concerns and are not
silently guessed in code.
