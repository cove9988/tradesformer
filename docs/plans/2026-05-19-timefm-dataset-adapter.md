# TimeFM Dataset Adapter Implementation Plan

> For Hermes: implement with strict TDD.

Goal: build a dataset adapter that turns weekly TimeFM CSV exports into leakage-safe rolling forecast samples.

Architecture: keep the adapter pure and lightweight. It should read weekly CSVs, validate required columns, generate rolling `(context, horizon)` windows, and return sample objects with target history, future target, historical covariates, and timestamp metadata. The first version should avoid model-framework coupling so it can feed evaluation, baseline models, or a later TimeFM wrapper.

Tech stack: Python, pandas, numpy, unittest.

---

## Task 1: define expected sample behavior in tests

Files:
- Create: `test/test_timefm_dataset.py`
- Read: `src/configure.json`, `src/timefm_features.py`

Steps:
1. Add a failing test for one weekly frame becoming the expected number of `(context, horizon)` windows.
2. Add a failing test proving sample alignment is leakage-safe: context ends immediately before forecast start.
3. Add a failing test proving invalid windows with NaNs are skipped when `dropna=True`.
4. Add a failing test proving directory loading sorts weekly files and expands configured `(context_window, forecast_horizon)` grids.

Verification:
- `./.venv/bin/python -m unittest discover -s test -p 'test_timefm_dataset.py'`
- Expected before implementation: FAIL because `src.timefm_dataset` does not exist yet.

## Task 2: implement the pure adapter

Files:
- Create: `src/timefm_dataset.py`
- Read: `src/data_processor.py`, `src/util/read_config.py`

Steps:
1. Add a `TimeFMSample` dataclass with context/future arrays and timestamp metadata.
2. Add a CSV loader that parses `time`, sorts rows, and validates required columns.
3. Add a rolling-window builder for one dataframe.
4. Add a file-level helper for one weekly CSV.
5. Add a directory-level helper that builds a dataset grid keyed by `(context_window, forecast_horizon)`.
6. Add a config-driven helper that uses `configure.json` defaults.

Verification:
- rerun the focused unittest module until green.

## Task 3: verify on real repo data shape

Files:
- Modify if needed: `src/README.md`

Steps:
1. If weekly split files do not exist, generate them with `python -m src.data_processor EURUSD`.
2. Run a small Python check that loads one configured dataset grid from `data/split/EURUSD/weekly/train`.
3. Confirm the adapter returns non-zero samples and sensible shapes for the configured windows.

Verification:
- sample counts are positive for at least one `(context, horizon)` pair
- first sample shapes match config expectations
