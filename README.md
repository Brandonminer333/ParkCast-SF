# ParkCast SF

## Branden Miner, Kayvan Zahiri, Temesghen Kahsay

## Streamlined EDA

This repo includes a small, repeatable EDA pipeline for SODA-style JSON endpoints (SFpark-style).

### How to run

1. Create the environment:
   - `conda env create -f environment.yml`
2. Start Jupyter and run the notebook cell in `eda.ipynb`.
   - The notebook calls `run_eda()` from `eda_pipeline.py`.

### What gets generated

After a run, artifacts are written under `outputs/eda/`:

- `missingness.csv` (per-column missing fraction)
- `dtypes.csv` (inferred dtypes after coercion)
- `numeric_describe.csv` (summary stats for numeric columns)
- `correlation.csv` (pairwise correlations among numeric columns)
- `plots/` (histograms for “availability/occupancy-like” columns, plus time-bucket mean plots when timestamps exist)
