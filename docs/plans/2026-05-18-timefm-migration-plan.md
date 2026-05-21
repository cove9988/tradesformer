# TimeFM Migration Plan

> For Hermes: use subagent-driven-development or Kanban execution to implement this plan task-by-task.

Goal: replace PPO as the primary prediction engine with a TimeFM/TimesFM-based forecasting pipeline, then drive trading decisions from forecast outputs rather than raw RL policy actions.

Architecture: keep the existing data-processing assets and weekly split workflow, but decouple forecasting, signal generation, and backtesting into separate modules. First establish a reproducible forecasting benchmark, then layer a deterministic trading policy over forecast outputs, and only later consider RL again for execution refinement.

Tech Stack: Python 3.10+, existing pandas/numpy pipeline, TimesFM, HuggingFace/PEFT only if fine-tuning is needed later, existing CSV datasets and repo-local evaluation logic.

---

## Phase 0: branch + planning baseline

### Task 0.1: create and verify the TimeFM branch

Objective: isolate the migration work from `main`.

Files:
- Modify: none

Step 1: create branch
Run: `git checkout -b TimeFM`
Expected: switched to a new branch named `TimeFM`

Step 2: verify branch status
Run: `git status --short --branch`
Expected: branch header shows `## TimeFM`

Step 3: commit planning artifacts later on this branch

---

## Phase 1: make the current PPO pipeline measurable

### Task 1.1: audit current training/eval entry points

Objective: document exactly how PPO is wired into prediction and decision-making today.

Files:
- Read: `src/training.py`
- Read: `src/eval.py`
- Read: `src/trading_gym_env.py`
- Create: `docs/plans/timefm-current-state-notes.md`

Step 1: trace where PPO is trained, loaded, and queried.
Step 2: document what the observation tensor contains.
Step 3: document action semantics, reward shaping, and where trading logic is embedded.
Step 4: write a short note identifying code that must remain reusable vs code that should be retired.

Verification:
- The note explicitly names the functions/classes to keep and to replace.

### Task 1.2: define baseline metrics and test windows

Objective: create a fair benchmark so the TimeFM migration can be compared to PPO.

Files:
- Create: `src/metrics.py`
- Modify: `src/eval.py` or create `src/eval_baseline.py`

Step 1: define forecast metrics:
- MAE / RMSE on future close or returns
- directional accuracy
- precision on thresholded moves

Step 2: define trading metrics:
- total return
- Sharpe
- max drawdown
- win rate
- profit factor
- trade count

Step 3: select a fixed evaluation set from existing weekly split files.

Verification:
- A single command can print a baseline metrics report for the current PPO pipeline.

---

## Phase 2: integrate TimeFM as a forecaster

### Task 2.1: add TimeFM dependency and environment notes

Objective: make the repo capable of loading a pretrained TimeFM model.

Files:
- Modify: `pyproject.toml`
- Modify: `README.md`
- Create: `docs/plans/timefm-setup-notes.md`

Step 1: add the minimum dependency set for TimeFM inference.
Step 2: document installation caveats for CPU / Apple Silicon / GPU.
Step 3: document the first supported model checkpoint and why it was chosen.

Verification:
- `poetry install` or the chosen environment install path completes with the new dependency set.
- Setup notes explain how to validate model loading.

### Task 2.2: create a dedicated forecaster module

Objective: isolate all TimeFM model loading and forecasting behind one interface.

Files:
- Create: `src/timefm_forecaster.py`
- Create: `test/test_timefm_forecaster.py`

Step 1: define a small interface like:
- `load_model(...)`
- `forecast_series(...)`
- `forecast_batch(...)`

Step 2: accept plain numpy/pandas sequences from existing weekly CSVs.
Step 3: return a structured result object containing:
- point forecast
- optional quantiles / uncertainty
- metadata (context length, horizon, model id)

Verification:
- Unit tests confirm the module can be instantiated and returns correctly shaped outputs.

### Task 2.3: map repo data into TimeFM inputs

Objective: bridge existing weekly CSV format to TimeFM-ready sequences.

Files:
- Create: `src/timefm_dataset.py`
- Create: `test/test_timefm_dataset.py`
- Read: `src/data_processor.py`

Step 1: choose the target forecast series format.
Recommended first target: close-price returns or normalized close series.

Step 2: define context window and forecast horizon presets.
Suggested initial presets:
- context: 96, 192, 288
- horizon: 12, 24

Step 3: provide helpers to extract rolling windows from weekly split files.

Verification:
- Tests prove a weekly CSV becomes valid `(context, horizon)` examples without shape drift or leakage.

---

## Phase 3: evaluate forecasting quality before trading

### Task 3.1: implement pure forecast evaluation

Objective: test TimeFM as a prediction engine before wiring in order logic.

Files:
- Create: `src/eval_timefm_forecast.py`
- Modify: `src/metrics.py`
- Create: `test/test_eval_timefm_forecast.py`

Step 1: run TimeFM over held-out weekly windows.
Step 2: score forecast quality with the agreed metrics.
Step 3: emit a machine-readable and human-readable summary.

Verification:
- One command prints aggregate metrics by symbol and by horizon.

### Task 3.2: compare TimeFM against simple baselines

Objective: avoid overclaiming based on raw model complexity.

Files:
- Modify: `src/eval_timefm_forecast.py`
- Create: `src/baselines.py`

Step 1: add at least these baselines:
- naive last-value
- simple momentum / drift
- existing repo transformer/PPO-derived proxy if practical

Step 2: compare across the same windows and horizons.

Verification:
- Report clearly shows whether TimeFM is actually adding predictive value.

---

## Phase 4: convert forecasts into trading signals

### Task 4.1: build a deterministic signal engine

Objective: turn forecasts into clear buy/sell/hold decisions without PPO.

Files:
- Create: `src/signal_engine.py`
- Create: `test/test_signal_engine.py`

Step 1: define signal inputs:
- expected return over horizon
- forecast slope
- uncertainty / quantile spread
- optional session filters

Step 2: define first-pass rules:
- buy if expected return > positive threshold and uncertainty acceptable
- sell if expected return < negative threshold and uncertainty acceptable
- otherwise hold

Step 3: define confidence gating to suppress low-edge trades.

Verification:
- Unit tests cover bullish, bearish, flat, and high-uncertainty cases.

### Task 4.2: build a simple backtest runner outside the PPO env

Objective: evaluate forecast-driven trades without the complexity of the RL environment.

Files:
- Create: `src/backtest_timefm.py`
- Create: `test/test_backtest_timefm.py`

Step 1: implement a deterministic backtest loop using:
- entry threshold
- SL/PT rules
- max open positions
- transaction cost assumptions

Step 2: reuse as much existing trade accounting logic as is sensible, but do not depend on PPO.

Verification:
- A single command can backtest one symbol/week set and print trading metrics.

---

## Phase 5: compare, decide, and clean up

### Task 5.1: produce a migration comparison report

Objective: summarize whether TimeFM should replace PPO in the main path.

Files:
- Create: `docs/plans/timefm-comparison-report.md`

Step 1: summarize forecast metrics.
Step 2: summarize backtest metrics.
Step 3: identify gaps, risks, and next experiments.
Step 4: recommend one of:
- replace PPO path
- keep both paths
- use TimeFM for forecasting and PPO only for execution later

Verification:
- The report contains a recommendation backed by measured results.

### Task 5.2: clean entry points and docs

Objective: make the repo understandable after the migration.

Files:
- Modify: `README.md`
- Modify: `src/README.md`
- Modify or deprecate: `src/training.py`, `src/eval.py`

Step 1: mark legacy PPO path clearly.
Step 2: document the new recommended path.
Step 3: add exact commands for setup, forecast eval, and backtest.

Verification:
- A new reader can tell what the primary path is in under two minutes.

---

## Initial Kanban decomposition

Use these execution lanes:
1. TimeFM repo audit and baseline metrics
2. TimeFM integration scaffolding
3. Forecast evaluation and baselines
4. Signal engine and backtest
5. Final comparison report and docs cleanup

Recommended dependency graph:
- lane 1 -> lane 3
- lane 2 -> lane 3
- lane 3 -> lane 4
- lane 4 -> lane 5

## Suggested first commands

- `git status --short --branch`
- `poetry install`
- `python -m src.eval_timefm_forecast`
- `python -m src.backtest_timefm`

## Notes

- Do not treat TimeFM as a direct trading policy.
- First prove forecast value before optimizing trade logic.
- Keep the first migration narrow: one symbol, one horizon family, one deterministic signal policy.
- Only revisit RL after TimeFM forecasting has a verified edge.