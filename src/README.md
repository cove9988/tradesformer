# TimeFM-first dataset pipeline

This repo now has a new TimeFM-first preprocessing direction with no backward-compatibility goals.

Primary design decisions:
- treat the weekly split as the dataset boundary
- keep raw OHLC for downstream execution/backtesting use
- build causal observation features only from information available up to each bar
- avoid whole-week normalization and indicator-heavy PPO-era feature engineering
- prefer relative-price, multi-horizon return, volatility, bar-structure, and session/context features

Current feature set exported by `src.data_processor`:
- close_rel_week_open
- high_rel_week_open
- low_rel_week_open
- open_rel_week_open
- returns_12
- returns_48
- returns_96
- returns_288
- realized_vol_48
- realized_vol_96
- realized_vol_288
- atr_rel
- close_vs_sma_288
- close_vs_sma_576
- sma_288_vs_sma_576
- bar_range
- bar_body
- close_loc_in_bar
- hour_sin
- hour_cos
- minute_sin
- minute_cos
- london_session
- ny_session
- overlap_session
- week_progress

Helper columns also exported for forecasting/backtesting utilities:
- log_return_1
- true_range
- atr
- sma_288
- sma_576
- iso_year
- iso_week

Raw prices preserved:
- open
- high
- low
- close

Main implementation files:
- `src/timefm_features.py`: feature definitions and validation
- `src/data_processor.py`: raw CSV -> weekly TimeFM dataset exporter
- `src/timefm_dataset.py`: weekly CSV -> rolling `(context, horizon)` TimeFM samples
- `src/configure.json`: TimeFM-first observation list and forecast settings

How to run:
- `python -m src.data_processor EURUSD`
- `python -m src.data_processor AUDUSD W ./data/raw/AUDUSD_M5.csv`

Output location:
- `./data/split/{symbol}/weekly/train/{symbol}_{iso_year}_{iso_week}.csv`

Dataset adapter:
- `build_timefm_dataset_from_config(data_directory)` loads weekly exports using:
  - `target_series`
  - `context_windows`
  - `forecast_horizons`
- returns a grid keyed by `(context_window, forecast_horizon)`
- each sample contains:
  - target history
  - dynamic covariate history
  - future target horizon
  - source file and timestamp boundaries

Notes:
- Weeks with fewer than 1440 bars are skipped.
- Weeks with missing observation features are skipped.
- The old weekly z-score path and PPO-era indicator assumptions are intentionally not preserved.
