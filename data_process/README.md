# Data processing demo

This folder documents how to run the TimeFM-first data processing pipeline.

## What the processor does

`src.data_processor`:
- reads a raw 5-minute FX CSV
- patches missing timestamps onto a 5-minute grid
- adds TimeFM-oriented causal features
- splits output into weekly files
- skips incomplete weeks or weeks with missing feature values

Output files are written to:
- `data/split/{symbol}/weekly/train/{symbol}_{iso_year}_{iso_week}.csv`

## Prerequisites

From the repo root:

```bash
cd /Users/s106916/github/tradesformer
```

If you want to use the local virtualenv:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install numpy pandas
```

## Run with the default raw file path

If your raw file follows the repo convention:
- `data/raw/EURUSD_M5.csv`
- `data/raw/AUDUSD_M5.csv`

then run:

```bash
python -m src.data_processor EURUSD
```

Example:

```bash
python -m src.data_processor GBPUSD
```

This automatically reads:
- `./data/raw/GBPUSD_M5.csv`

## Run with an explicit input file

If you want to point to a specific raw CSV:

```bash
python -m src.data_processor AUDUSD W ./data/raw/AUDUSD_M5.csv
```

Notes:
- the middle `W` argument is currently only a placeholder for the CLI shape
- the script really uses the symbol and the input CSV path

## Run with the repo virtualenv

If you are using the local `.venv` created in this repo:

```bash
./.venv/bin/python -m src.data_processor EURUSD
```

## Example end-to-end

```bash
mkdir -p data/log
./.venv/bin/python -m src.data_processor EURUSD
```

After it finishes, you should see weekly files under:

```bash
data/split/EURUSD/weekly/train/
```

For example:
- `data/split/EURUSD/weekly/train/EURUSD_2024_46.csv`

## What is inside each weekly file

Each exported weekly CSV keeps raw prices:
- `open`
- `high`
- `low`
- `close`

And includes TimeFM-oriented features such as:
- `close_rel_week_open`
- `returns_12`
- `returns_48`
- `returns_96`
- `returns_288`
- `realized_vol_48`
- `realized_vol_96`
- `realized_vol_288`
- `atr_rel`
- `close_vs_sma_288`
- `close_vs_sma_576`
- `week_progress`
- session/time context features

## Common issues

### 1. Missing log directory

If logging fails, create the log folder:

```bash
mkdir -p data/log
```

### 2. Missing Python packages

Install the minimum required packages:

```bash
pip install numpy pandas
```

### 3. Raw CSV not found

Make sure the input exists, for example:

```bash
ls data/raw/EURUSD_M5.csv
```

## Next step after preprocessing

To turn weekly exports into rolling forecast samples, use the dataset adapter in:
- `src/timefm_dataset.py`

Example Python snippet:

```python
from src.timefm_dataset import build_timefm_dataset_from_config

grid = build_timefm_dataset_from_config('./data/split/EURUSD/weekly/train')
print(grid.keys())
```
