# PRDetect GCN/R-GCN

The original GCN remains the default. Use a separate graph directory and checkpoint
directory for experiments:

```text
python building_graph.py --input_dir /kaggle/input/PRDetect/original_text --output_dir /kaggle/working/graph_data
python add_edge_type.py --pickle-dir /kaggle/working/graph_data --json-dir /kaggle/input/PRDetect/original_text --files hc3_train hc3_val hc3_test
python gcn.py --model rgcn --seed 2024 --dataset hc3 --data_dir graph_data --out_dir model
python test.py --model rgcn --seed 2024 --dataset hc3 --file hc3_test --data_dir graph_data --save
python detect.py --model rgcn --dataset hc3 --seed 2024 --text "text to classify"
```

For the GCN baseline, omit `--model rgcn`. R-GCN requires `all_edge_type` in every
pickle. `--add_reverse 1` appends reverse relations, and `--use_labels 0` collapses
all dependency labels to one relation. The implementation does not rerun RoBERTa when
`add_edge_type.py` augments an existing pickle.
