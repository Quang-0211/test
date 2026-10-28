# Kaggle setup

1. Attach a Kaggle Dataset containing the PRDetect checkout, `roberta-base/`, and
   original/perturbed JSON files. Enable GPU; internet is optional if all assets are
   attached.
2. From `/kaggle/working/PRDetect`, install the runtime dependencies:

```text
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. If graph pickles are absent, run `building_graph.py` once with output under
   `/kaggle/working/graph_data`; otherwise run `add_edge_type.py` against the attached
   pickles. Keep intermediate pickles in `/kaggle/working` or publish them as a Kaggle
   Dataset for later sessions.
4. Run `python kaggle/run_matrix.py --only A0,B0,B2 --seeds 2024` as a smoke test,
   then expand to all seeds/configurations.
