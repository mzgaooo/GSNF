# GSNF Classification

This repository contains the cleaned GSNF implementation for irregular EHR time-series binary classification.

Supported experiment datasets:

- `physionet12`: PhysioNet 2012 with 37 dynamic variables. The split seed is fixed to `290`.
- `p12`: full PhysioNet 2012 processed sequence format. The split seed is fixed to `155`.
- `P19`: PhysioNet 2019 processed sequence format.
- `eICU`: eICU mortality classification with preprocessed CSV files.
- `MIMIC4`: MIMIC-IV mortality classification with preprocessed CSV files.

Datasets are not included in this repository. Put data under `data/` or pass an external root with `--data-dir`.

Expected data layout:

```text
data/
  PhysioNet12/
    processed/p12_data.csv
    processed/p12_labels.csv
  IrregularTimeSeriesDatasets/
    P12data/processed_data/PTdict_list.npy
    P12data/processed_data/arr_outcomes.npy
    P19data/processed_data/PTdict_list.npy
    P19data/processed_data/arr_outcomes.npy
  eicu/processed/
    eicu_data.csv
    eicu_labels.csv
  mimic4/processed/
    mimic4_full_dataset.csv
    mortality_labels.csv
```

Example command:

```bash
python main.py \
  --data physionet12 \
  --data-dir /path/to/data \
  --batch-size 50 \
  --time-max 2880 \
  --hidden-dim 128 \
  --hidden-layers 3 \
  --flow-layers 4 \
  --time-net TimeTanh \
  --time-hidden-dim 8 \
  --gcn-hidden-dim 64 \
  --gnn-flow-conv basic \
  --classifier-head gsnf \
  --classifier-input z0 \
  --ratio-ce 1000 \
  --ratio-itg 0.1 \
  --ratio-rtg 0.1 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --early-stop-metric auroc
```
