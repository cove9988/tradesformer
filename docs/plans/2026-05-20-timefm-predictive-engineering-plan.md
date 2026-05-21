# TimeFM Predictive Engineering Plan

> For Hermes: use subagent-driven-development or Kanban execution to implement this plan task-by-task.

Goal: turn the repo's current weekly TimeFM-ready EURUSD data into a measured forecasting pipeline, then train a lightweight decision/calibration layer on top of TimeFM outputs before introducing any new trading policy logic.

Architecture: keep `src/timefm_features.py`, `src/data_processor.py`, and `src/timefm_dataset.py` as the canonical data layer. Add a split manifest, forecast target helpers, a TimeFM forecaster wrapper, a forecast artifact pipeline, a calibration trainer, and a pure evaluation harness. Keep legacy PPO training isolated and do not reuse it as the primary TimeFM training entry point.

Tech Stack: Python 3.10, pandas, numpy, scikit-learn, current weekly CSV dataset, a verified TimeFM/TimesFM inference package, optional PEFT/LoRA only if later fine-tuning is justified by measured results.

---

## Current repo reality to plan around

These facts were verified from the current repo before writing this plan:

- `data/split/EURUSD/weekly/train` currently contains 138 weekly CSV files.
- The files span `EURUSD_2022_13.csv` through `EURUSD_2024_46.csv`.
- Each weekly file contains 1440 five-minute bars.
- `src/configure.json` defines:
  - `target_series = log_return_1`
  - `context_windows = [96, 192, 288]`
  - `forecast_horizons = [12, 24]`
  - 26 causal observation features.
- Approximate rolling-sample counts from the current train data are:
  - `(96, 12)` -> 183,954 samples
  - `(96, 24)` -> 182,298 samples
  - `(192, 12)` -> 170,706 samples
  - `(192, 24)` -> 169,050 samples
  - `(288, 12)` -> 157,458 samples
  - `(288, 24)` -> 155,802 samples
- `src/timefm_dataset.py` already builds leakage-safe rolling samples.
- There is currently no populated `eval` split for EURUSD.
- `src/training.py` and `src/eval.py` are PPO-era entry points and still depend on legacy config assumptions such as `backward_window` / `observation_list`; they should be treated as legacy, not as the starting point for TimeFM training.

This means the first job is not “train a big new model immediately”. The first job is to create a clean supervised forecasting path and a deterministic way to train on chronological splits using the data we already have.

---

## What we should develop first

In priority order:

1. A deterministic train/validation/test split manifest for weekly files.
2. A target-definition layer for forecast training and evaluation.
3. A TimeFM forecaster wrapper with batch inference.
4. A forecast artifact pipeline that writes predictions to disk.
5. A lightweight trained calibration layer on top of TimeFM outputs.
6. A forecasting evaluation harness with strong naive baselines.
7. Only after that: signal policy and backtest integration.
8. Only after that: optional TimeFM fine-tuning.

---

## Training strategy recommendation

### Recommendation: do not start by fine-tuning TimeFM itself

With the current verified dataset, we have one symbol (`EURUSD`) and 138 weekly files. That is enough to build a serious supervised benchmark and to train a small downstream model, but it is not yet a strong reason to start with full foundation-model fine-tuning.

The recommended training sequence is:

1. Use pretrained TimeFM as the base forecaster.
2. Train a lightweight calibration / decision model on current train weeks only.
3. Validate on a held-out validation block.
4. Test once on a final untouched test block.
5. Consider TimeFM fine-tuning only if:
   - pretrained TimeFM shows clear directional signal but poor scale calibration, or
   - the calibrator still leaves significant systematic error, and
   - we are ready to add the dependency and runtime complexity.

### What “training with current training data” should mean initially

The first trained component should be a small supervised model using the existing weekly samples and TimeFM outputs, for example:

- regression target: realized cumulative log return over the next horizon
- classification target: whether the next-horizon return exceeds a threshold
- auxiliary target: direction only (`up`, `down`, `flat`)

Recommended first trained model types:

- `sklearn.linear_model.Ridge`
- `sklearn.linear_model.LogisticRegression`
- `sklearn.ensemble.RandomForestRegressor` only if a linear calibrator is too weak

Start simple. The current repo already depends on `scikit-learn`, so this avoids unnecessary complexity while still giving us a genuinely trained predictive layer.

---

## Proposed chronological split for current EURUSD data

Because there is no real eval split yet, create one from the verified weekly files.

Recommended first split:

- Train: `2022_13` through `2024_34` -> 126 weeks
- Validation: `2024_35` through `2024_40` -> 6 weeks
- Test: `2024_41` through `2024_46` -> 6 weeks

Rules:

- Never shuffle windows across time.
- Windows from the same source week must stay inside one split.
- Hyperparameter selection happens on validation only.
- Test is used once for the final measured report.

Later, add walk-forward evaluation, but do not block the first milestone on it.

---

## Phase 0: establish the predictive-engineering contract

### Task 0.1: create a split manifest generator

Objective: make training reproducible by defining exactly which weekly CSV belongs to train, validation, and test.

Files:
- Create: `src/timefm_split.py`
- Create: `test/test_timefm_split.py`
- Create: `data/split/EURUSD/weekly/manifest.json`

Steps:
1. Add a helper that scans `data/split/{symbol}/weekly/train/*.csv`.
2. Parse `symbol`, `iso_year`, and `iso_week` from filenames.
3. Build a deterministic chronological manifest.
4. Write the first fixed split using the 126/6/6 weekly split above.
5. Ensure later code loads splits from the manifest instead of hardcoding filenames.

Verification:
- `./.venv/bin/python -m unittest discover -s test -p 'test_timefm_split.py'`
- `./.venv/bin/python -m src.timefm_split --symbol EURUSD --write-manifest`
- Expected: a manifest exists at `data/split/EURUSD/weekly/manifest.json` with non-overlapping train/val/test week lists.

### Task 0.2: add a dataset audit command

Objective: make the current TimeFM dataset measurable before any model work.

Files:
- Create: `src/timefm_data_audit.py`
- Create: `test/test_timefm_data_audit.py`

Steps:
1. Add a command that reads the manifest and counts files, bars, and rolling samples per `(context, horizon)` pair.
2. Report NaN rates for `target_series` and configured observation columns.
3. Report min/max timestamp coverage per split.
4. Print a compact table and optionally write JSON to `data/reports/timefm_data_audit_EURUSD.json`.

Verification:
- `./.venv/bin/python -m src.timefm_data_audit --symbol EURUSD`
- Expected: counts align with the manifest and current data shape.

---

## Phase 1: make forecast targets explicit

### Task 1.1: define horizon-level targets for training and evaluation

Objective: stop relying only on raw `log_return_1` and define the actual prediction targets we care about.

Files:
- Create: `src/timefm_targets.py`
- Create: `test/test_timefm_targets.py`
- Modify: `src/timefm_dataset.py`
- Modify: `src/configure.json`

Steps:
1. Keep `log_return_1` as the sequence-level base target used for rolling samples.
2. Add helpers that derive horizon-level labels from `target_future`, including:
   - `future_return_sum`
   - `future_direction`
   - `future_abs_move`
3. Make these derived targets available in the sample metadata or in a downstream labeling helper.
4. Add config entries for the primary supervised objective, for example:
   - `forecast_objective = cumulative_log_return`
   - `direction_thresholds = {12: ..., 24: ...}`

Verification:
- `./.venv/bin/python -m unittest discover -s test -p 'test_timefm_targets.py'`
- Expected: tests prove labels are derived only from the future window and are horizon-consistent.

### Task 1.2: extend the dataset loader for split-aware training

Objective: make the dataset adapter usable directly by training and evaluation jobs.

Files:
- Modify: `src/timefm_dataset.py`
- Create: `test/test_timefm_dataset_splits.py`

Steps:
1. Add helpers to load samples for `train`, `val`, or `test` based on the manifest.
2. Preserve `source_file`, `iso_year`, and `iso_week` metadata per sample.
3. Add optional downsampling controls so quick experiments can run on a subset.
4. Add an iterator or generator mode for batch inference to avoid loading all windows into memory at once.

Verification:
- `./.venv/bin/python -m unittest discover -s test -p 'test_timefm_dataset*.py'`
- Expected: split-aware loading never mixes source weeks across train/val/test.

---

## Phase 2: integrate TimeFM as the forecasting engine

### Task 2.1: perform a minimal TimeFM loader spike

Objective: verify the exact package, checkpoint, and runtime constraints before building repo abstractions.

Files:
- Create: `src/timefm_loader_spike.py`
- Create: `docs/plans/timefm-loader-spike-notes.md`
- Modify: `pyproject.toml`

Steps:
1. Confirm the exact TimeFM/TimesFM Python package and supported checkpoint to use in this repo.
2. Add only the minimum dependency set required for batch inference.
3. Write a tiny spike script that loads the model and runs one forecast on one sample.
4. Document Apple Silicon / CPU / GPU caveats in the spike notes.

Verification:
- `poetry install`
- `./.venv/bin/python -m src.timefm_loader_spike`
- Expected: one sample forecast completes and prints the returned horizon shape.

### Task 2.2: create the production forecaster wrapper

Objective: isolate all TimeFM interaction behind one stable repo interface.

Files:
- Create: `src/timefm_forecaster.py`
- Create: `test/test_timefm_forecaster.py`

Steps:
1. Define a small interface:
   - `load_model(...)`
   - `forecast_sample(...)`
   - `forecast_batch(...)`
2. Return a structured forecast result containing:
   - predicted future sequence
   - optional quantiles / uncertainty
   - model id
   - context length
   - forecast horizon
3. Ensure the interface accepts `TimeFMSample` objects directly.
4. Keep file-system concerns out of this module.

Verification:
- `./.venv/bin/python -m unittest discover -s test -p 'test_timefm_forecaster.py'`
- Expected: wrapper tests pass with either a stub model or the verified package.

### Task 2.3: create a forecast artifact pipeline

Objective: persist TimeFM predictions so downstream training and evaluation do not rerun inference unnecessarily.

Files:
- Create: `src/run_timefm_forecasts.py`
- Create: `test/test_run_timefm_forecasts.py`
- Create: `data/predictions/.gitkeep`

Steps:
1. Load a manifest-defined split.
2. Run batch forecasts for each configured `(context, horizon)` pair.
3. Write forecast artifacts to a deterministic path such as:
   - `data/predictions/timefm/EURUSD/{split}/context_{c}_horizon_{h}.parquet`
4. Include realized future targets alongside predictions so later trainers and evaluators can read one artifact set.

Verification:
- `./.venv/bin/python -m src.run_timefm_forecasts --symbol EURUSD --split train`
- Expected: forecast artifacts are written for each configured context/horizon pair.

---

## Phase 3: train the first learned component on current data

### Task 3.1: train a lightweight calibration model on top of TimeFM forecasts

Objective: use current training data to train a simple learned layer that maps TimeFM outputs into better-scaled predictive decisions.

Files:
- Create: `src/timefm_calibrator.py`
- Create: `src/train_timefm_calibrator.py`
- Create: `test/test_timefm_calibrator.py`

Steps:
1. Read train-split forecast artifacts.
2. Build compact training features from TimeFM outputs, for example:
   - predicted cumulative return
   - predicted slope across the forecast horizon
   - quantile spread or uncertainty proxy
   - recent realized volatility from the last context bars
   - session/time-context summaries
3. Train two first-pass models:
   - regression model for realized cumulative return
   - classification model for positive/negative direction
4. Fit only on `train`.
5. Select thresholds and regularization only on `val`.
6. Save the fitted model and feature schema under `data/models/timefm_calibrator/`.

Verification:
- `./.venv/bin/python -m src.train_timefm_calibrator --symbol EURUSD --context 96 --horizon 12`
- Expected: model artifacts, validation metrics, and chosen thresholds are saved.

### Task 3.2: explicitly defer foundation-model fine-tuning until there is evidence

Objective: prevent premature complexity.

Files:
- Create: `docs/plans/timefm-finetune-gate.md`

Steps:
1. Define hard gates for when fine-tuning is allowed.
2. Require that the pretrained TimeFM + calibrator path has already been measured.
3. Require a documented failure mode, such as strong directional edge but poor amplitude calibration.
4. Require an implementation note on expected package/runtime cost before PEFT/LoRA is added.

Verification:
- The gate note contains objective go/no-go criteria rather than vague “maybe later”.

---

## Phase 4: measure predictive quality before any trading logic

### Task 4.1: build a forecasting evaluation harness

Objective: prove whether TimeFM plus calibration is actually useful as a prediction engine.

Files:
- Create: `src/eval_timefm_pipeline.py`
- Create: `src/metrics_timefm.py`
- Create: `test/test_eval_timefm_pipeline.py`

Steps:
1. Score raw TimeFM forecasts.
2. Score calibrated forecasts.
3. Report metrics by split, context window, and horizon.
4. Include at least:
   - MAE
   - RMSE
   - directional accuracy
   - precision / recall for thresholded moves
   - calibration error for sign and magnitude buckets
5. Write both console and machine-readable reports.

Verification:
- `./.venv/bin/python -m src.eval_timefm_pipeline --symbol EURUSD --split val`
- Expected: a report is emitted for every configured context/horizon pair.

### Task 4.2: compare against strong simple baselines

Objective: avoid overclaiming just because TimeFM is a foundation model.

Files:
- Create: `src/baselines_timefm.py`
- Modify: `src/eval_timefm_pipeline.py`
- Create: `test/test_baselines_timefm.py`

Steps:
1. Add at least these baselines:
   - zero-return baseline
   - last-return carry-forward baseline
   - rolling-mean baseline
   - simple drift baseline
2. Evaluate all baselines on the same manifest-defined windows.
3. Report when TimeFM loses, not just when it wins.

Verification:
- Final evaluation output clearly ranks TimeFM, calibrated TimeFM, and all baselines on the same splits.

---

## Phase 5: promote the predictive layer into a trading-facing interface

### Task 5.1: create a deterministic signal policy from calibrated forecasts

Objective: convert predictive outputs into a stable buy/sell/hold decision policy without bringing PPO back in.

Files:
- Create: `src/timefm_signal_policy.py`
- Create: `test/test_timefm_signal_policy.py`

Steps:
1. Define signal inputs from the calibrated forecast outputs.
2. Add thresholding rules for long, short, and hold.
3. Add uncertainty gating and minimum-edge filters.
4. Ensure the policy is deterministic and config-driven.

Verification:
- `./.venv/bin/python -m unittest discover -s test -p 'test_timefm_signal_policy.py'`
- Expected: bullish, bearish, flat, and low-confidence cases are all covered.

### Task 5.2: add a pure backtest runner for the predictive path

Objective: measure the forecasting path without PPO environment coupling.

Files:
- Create: `src/backtest_timefm_pipeline.py`
- Create: `test/test_backtest_timefm_pipeline.py`

Steps:
1. Read calibrated predictions or signal outputs.
2. Apply deterministic entry/exit rules.
3. Use raw OHLC columns from the weekly files for trade accounting.
4. Report total return, Sharpe, drawdown, win rate, profit factor, and trade count.

Verification:
- `./.venv/bin/python -m src.backtest_timefm_pipeline --symbol EURUSD --split test`
- Expected: one full predictive-path backtest report is produced with no PPO dependency.

---

## Phase 6: only then consider fine-tuning TimeFM

### Task 6.1: add an optional fine-tuning experiment path

Objective: provide a place for adapter tuning only after the earlier stages are green.

Files:
- Create: `src/train_timefm_adapter.py`
- Create: `test/test_train_timefm_adapter.py`
- Modify: `pyproject.toml`
- Create: `docs/plans/timefm-finetuning-notes.md`

Steps:
1. Reuse the manifest-defined splits and forecast targets.
2. Start with one `(context, horizon)` pair, not the whole grid.
3. Log training and validation curves.
4. Compare fine-tuned results directly against pretrained TimeFM + calibrator.
5. Kill this path if it does not beat the simpler system clearly.

Verification:
- A single experiment report shows whether fine-tuning improved forecasting quality enough to justify the added complexity.

---

## Recommended first training run with current data

Once Phases 0 through 3 exist, the first real training run should look like this:

1. Generate the manifest.
2. Audit the dataset.
3. Run pretrained TimeFM forecasts for `train`, `val`, and `test`.
4. Train a calibrator on `train` using `(context=96, horizon=12)` first.
5. Tune on `val`.
6. Freeze the best settings.
7. Evaluate once on `test`.
8. Repeat the same process for `(96,24)`, `(192,12)`, `(192,24)`, `(288,12)`, `(288,24)`.
9. Compare all six settings and select one default predictive engine.

Why start with `(96, 12)`:

- shortest context means fastest experimentation
- shortest horizon gives the quickest feedback loop
- it still has the largest current sample count

---

## Deliverables that should exist when this plan is complete

- reproducible split manifest
- dataset audit command and report
- explicit target-definition module
- TimeFM loader spike notes
- production `timefm_forecaster` wrapper
- forecast artifact pipeline
- trained calibrator artifacts
- forecast evaluation report with baselines
- deterministic signal policy
- predictive-path backtest report
- optional fine-tuning gate and experiment path

---

## Success criteria

This plan succeeds when all of the following are true:

- A new contributor can train the predictive path without touching PPO code.
- Train/validation/test splits are deterministic and chronological.
- TimeFM forecasts are saved as reusable artifacts.
- At least one calibrated TimeFM configuration beats naive baselines on validation and test.
- A deterministic signal layer can consume the predictive outputs.
- Any decision to fine-tune TimeFM is based on measured evidence, not hype.

---

## Suggested first execution commands after this plan

- `git status --short --branch`
- `./.venv/bin/python -m unittest discover -s test -p 'test_timefm_dataset.py'`
- `./.venv/bin/python -m src.timefm_split --symbol EURUSD --write-manifest`
- `./.venv/bin/python -m src.timefm_data_audit --symbol EURUSD`
- `./.venv/bin/python -m src.timefm_loader_spike`
- `./.venv/bin/python -m src.run_timefm_forecasts --symbol EURUSD --split train`
- `./.venv/bin/python -m src.train_timefm_calibrator --symbol EURUSD --context 96 --horizon 12`
- `./.venv/bin/python -m src.eval_timefm_pipeline --symbol EURUSD --split val`

## Bottom line

For this repo today, TimeFM should enter as a forecasting foundation model plus a trained calibration layer, not as a drop-in replacement for PPO and not as an immediate fine-tuning project. The current weekly EURUSD data is already enough to build and train that predictive-engineering stack in a disciplined, measurable way.
