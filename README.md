# tradesformer

Experimental FX research repo focused on TimeFM/TimesFM-style forecasting and simple, honest backtests on 5-minute data.

This branch is no longer centered on the older PPO trading-agent approach. The active direction is:
- weekly-split, causal dataset construction
- TimesFM 2.5 volatility forecasting
- multi-pair LoRA fine-tuning experiments
- economic evaluation against simple baselines
- explicit go/no-go checks before claiming trading value

## Current branch status

This branch has a clear research conclusion so far:
- TimesFM volatility forecasts do appear to contain real predictive signal versus simple baselines.
- But on the current intraday FX monetization path tested here, that forecast edge does **not** become a useful trading/sizing edge.

Most important findings from the current work:
1. Zero-shot and fine-tuned TimesFM can beat naive volatility baselines on forward realized-vol metrics.
2. A simple HAR-style linear volatility model is also strong, so model wins must be judged against stronger baselines than a lagged EWMA alone.
3. In the current 5-pair intraday momentum setup, inverse-vol / vol-target sizing is a **no-go**: constant position sizing beats the tested vol-target overlays, even before realistic costs.
4. Honest interpretation: the model may predict volatility better, but the current strategy channel does not convert that into economic value.

So this branch is best read as a forecasting-and-evaluation research branch, not a production trading system.

## What this repo is testing

The active question is not “can a transformer trade FX direction directly?”

It is closer to:
- can TimesFM forecast short-horizon realized volatility on FX 5-minute data?
- does multi-pair fine-tuning improve that forecast?
- does any forecast improvement survive stronger baselines and cost-aware backtests?

## Data pipeline

The repo now uses a TimeFM-first weekly dataset pipeline with no backward-compatibility goal for the older PPO-era feature design.

Design choices:
- treat each weekly CSV as the dataset boundary
- keep raw OHLC for downstream execution/backtesting use
- use only causal features available up to each bar
- avoid whole-week normalization and heavy indicator stacks
- prefer relative-price, return, realized-vol, bar-structure, and session/context features

Primary exported features include:
- week-open-relative prices
- multi-horizon returns
- realized volatility windows
- ATR-relative regime features
- moving-average relative features
- bar-structure features
- session flags and weekly progress

Helper columns retained for research utilities include:
- `log_return_1`
- `true_range`
- `atr`
- `sma_288`
- `sma_576`
- `iso_year`
- `iso_week`

See also:
- `src/README.md`
- `src/data_processor.py`
- `src/timesfm_features.py`
- `src/timesfm_dataset.py`
- `src/configure.json`

## Repository layout

- `src/data_processor.py` — raw CSV to weekly split exporter
- `src/timesfm_features.py` — causal feature definitions and validation
- `src/timesfm_dataset.py` — rolling `(context, horizon)` dataset construction
- `src/timesfm_forecaster.py` — TimesFM wrapper / loading helpers
- `src/timesfm_vol_eval.py` — core realized-vol forecasting evaluation
- `src/timesfm_vol_trade.py` — trade vol against implied-vol proxy harness
- `src/timesfm_vol_finetune.py` — single-pair volatility fine-tuning
- `src/timesfm_vol_finetune_multipair.py` — pooled multi-pair fine-tuning
- `src/timesfm_vol_finetune_multipair_econ.py` — pooled fine-tuning with econIC-based checkpoint selection
- `src/timesfm_multipair_forward_eval.py` — 2025-26 forward matrix across adapters/pairs
- `src/har_baseline_eval.py` — HAR baseline through the same evaluation harness
- `src/voltarget_backtest.py` — minimal vol-target sizing ladder / go-no-go test
- `test/` — unit tests for datasets, evaluation, and experiment integrity guards

## Environment

This repo is managed with Poetry.

Install dependencies:

```bash
poetry install
```

The branch currently relies on the Hugging Face Transformers path for TimesFM 2.5, not the retired TimesFM v1 JAX/PAX path.

Key dependencies from `pyproject.toml` include:
- Python 3.10+
- numpy
- pandas
- scikit-learn
- torch
- stable-baselines3
- transformers
- accelerate
- peft

## Typical workflow

### 1) Build weekly splits

```bash
python -m src.data_processor EURUSD
python -m src.data_processor AUDUSD W ./data/raw/AUDUSD_M5.csv
```

Output goes to:

```text
data/split/{symbol}/weekly/train/{symbol}_{iso_year}_{iso_week}.csv
```

### 2) Evaluate volatility forecasting

```bash
PYTHONPATH=. python -m src.timesfm_vol_eval --context-len 512 --horizons 48 96 --window 48
PYTHONPATH=. python -m src.timesfm_vol_eval --backend fake
```

### 3) Trade the forecast against a simple implied-vol proxy

```bash
PYTHONPATH=. .venv/bin/python3.12 -m src.timesfm_vol_trade --backend fake --split test
```

### 4) Fine-tune pooled multi-pair volatility adapters

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  .venv/bin/python3.12 -m src.timesfm_vol_finetune_multipair_econ \
  --pairs EURUSD GBPUSD USDCAD USDCHF USDJPY \
  --context-len 512 --horizon-len 96 --window 48 \
  --val-weeks-per-pair 21 --num-samples 16000 --epochs 10 \
  --batch-size 16 --lr 5e-4 --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
  --output-dir data/adapters/multipair_vol_h96_econ
```

### 5) Run forward comparison matrix

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  .venv/bin/python3.12 -m src.timesfm_multipair_forward_eval --device mps
```

### 6) Run stronger baseline / monetization checks

```bash
PYTHONPATH=. .venv/bin/python3.12 -m src.har_baseline_eval
PYTHONPATH=. .venv/bin/python3.12 -m src.voltarget_backtest
```

## Important experiment files

### Core forecasting
- `src/timesfm_vol_eval.py`
- `src/timesfm_vol_trade.py`
- `src/timesfm_breakout_vol_gate.py`

### Fine-tuning and forward tests
- `src/timesfm_vol_finetune.py`
- `src/timesfm_vol_finetune_multipair.py`
- `src/timesfm_vol_finetune_multipair_econ.py`
- `src/timesfm_multipair_forward_eval.py`

### Honest benchmark / falsification scripts
- `src/har_baseline_eval.py`
- `src/voltarget_backtest.py`

### Tests guarding experiment credibility
- `test/test_timesfm_vol_eval.py`
- `test/test_timesfm_vol_finetune.py`
- `test/test_timesfm_vol_finetune_multipair.py`
- `test/test_timesfm_vol_finetune_multipair_econ.py`
- `test/test_timesfm_vol_trade.py`

## Output artifacts

Common output locations used by this branch:
- `data/predictions/eval/`
- `data/predictions/forward/`
- `data/adapters/`

Examples generated by the current work include:
- `data/predictions/forward/multipair_test3_econ_matrix.json`
- `data/predictions/forward/har_baseline_matrix.json`
- `data/predictions/forward/voltarget_config.json`

## Research standards for this branch

This branch is intentionally strict about honesty:
- do not treat weak baseline wins as proof of trading value
- separate forecasting skill from monetization skill
- test leakage boundaries explicitly
- prefer negative results over optimistic stories
- require cost-aware comparisons before making any trading claim

## Limitations

Important limitations of the current setup:
- no real options/implied-vol feed; `timesfm_vol_trade.py` uses an EWMA-based implied-vol proxy
- no interest-rate panel, so true FX carry is not tested here
- current sizing backtest is an intraday momentum/mean-reversion proxy, not a full portfolio construction stack
- strong forecasting metrics do not imply live profitability

## Legacy material

Older PPO/transformer trading-agent material still exists in the repository history and some subdirectories, but it does not represent the main direction of this branch anymore.

If you want the feature-engineering and weekly pipeline details, start with `src/README.md`.
