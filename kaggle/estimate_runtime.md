# Runtime estimate

This is planning guidance, not a measurement. The matrix has 9 configurations and 5
seeds, with one train and multiple test datasets per completed configuration. Prioritize
A0, B0, and B2 on Replace/Delete first, then cross-dataset and short-text evaluations,
then A1/B1/C1/D ablations. Graph construction and RoBERTa encoding should be performed
once and cached in `/kaggle/working/graph_data`; resume support skips completed
configuration/seed marker files.
