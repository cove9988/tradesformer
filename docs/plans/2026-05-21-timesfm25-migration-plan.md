# TimesFM 2.5 + Transformers + LoRA Migration Plan

> For Hermes: follow strict TDD. Write failing tests first, then implement the minimal code to pass. The v1 `timesfm` / JAX / PAX / Praxis path is being retired for this project.

Goal: replace the repo's current TimeFM v1 inference path with a TimesFM 2.5 Transformers-based forecasting and LoRA fine-tuning path that runs on a normal PyTorch/HuggingFace stack.

Architecture: keep the existing weekly CSV + split-manifest + leakage-safe rolling-window data layer. Replace the model/runtime layer with a Transformers-based TimesFM 2.5 wrapper for inference. Add a separate fine-tuning/training adapter layer for LoRA instead of overloading the inference wrapper. Preserve the current forecast artifact pipeline shape so downstream evaluation/backtesting code can keep reading JSONL forecasts.

Tech stack: Python 3.10+, torch, transformers, peft, accelerate, pandas, numpy, existing weekly CSV dataset, HuggingFace model checkpoints.

---

## Current repo reality we are migrating away from

Verified current v1-specific integration points:

1. `src/timefm_forecaster.py`
   - hardcodes v1 defaults via `TIMEFM_V1_DEFAULTS`
   - supports `backend_name='timesfm'` by importing `timesfm.TimesFm`
   - calls `load_from_checkpoint(repo_id=...)`
   - calls `.forecast([context], freq=[0])`
   - still includes fake backend support via `FakeTimesFMBackend`

2. `src/timefm_loader_spike.py`
   - defaults to `backend_name='fake-timesfm'`
   - defaults to `model_id='google/timesfm-1.0-200m'`
   - assumes `load_model()` returns the old v1 wrapper handle

3. `src/run_timefm_forecasts.py`
   - reads split samples and writes JSONL artifacts
   - imports `load_model`, `forecast_batch`, `TimeFMModelHandle`, `TimeFMForecastResult`
   - defaults to `backend_name='fake-timesfm'`
   - defaults to `model_id='google/timesfm-1.0-200m'`

4. `test/test_timefm_phase2.py`
   - includes v1-specific tests for:
     - `timesfm.TimesFm(**kwargs)`
     - `load_from_checkpoint(repo_id=...)`
     - `.forecast([context], freq=[0])`
   - also includes still-useful fake-backend and JSONL artifact tests

5. `pyproject.toml`
   - still declares `timesfm = "^1.0.0"`
   - does not declare `transformers`, `peft`, or `accelerate`

6. `src/timefm_dataset.py`
   - is still valuable and should stay the canonical rolling-window data layer
   - already produces leakage-safe `TimeFMSample` objects with:
     - `target_context`
     - `target_future`
     - `context_window`
     - `forecast_horizon`
     - timestamp boundaries
     - derived target metadata
   - currently stores `dynamic_covariates`, but the current model wrapper ignores them

7. Current environment reality
   - current `.venv` has none of: `torch`, `transformers`, `peft`, `accelerate`, `timesfm`
   - package index availability was verified for `torch`, `transformers`, `peft`, `accelerate`
   - v1 runtime is deprecated for this migration

---

## Migration decision

We are explicitly dumping the v1 path.

That means:
- no more primary dependency on `timesfm==1.0.0`
- no more JAX/PAX/Praxis-based loading path in `src/timefm_forecaster.py`
- no more `google/timesfm-1.0-200m` as the default runtime target
- no more tests asserting `TimesFm(...).load_from_checkpoint(...)`

Instead, the repo should target a TimesFM 2.5 path based on one of these practical interfaces:
- HuggingFace Transformers path:
  - `transformers.TimesFm2_5ModelForPrediction.from_pretrained(...)`
- or the newer PyTorch wrapper path documented in repo-local references:
  - `timesfm.TimesFM_2p5_200M_torch.from_pretrained(...)`

For this project migration, prefer the Transformers path for training/fine-tuning because it aligns with the official Google finetuning example and standard LoRA tooling.

---

## File-by-file migration map

### Keep mostly as-is

1. `src/timefm_dataset.py`
   - keep as the canonical rolling-window/sample builder
   - may need small additions for torch dataset conversion helpers
   - do not delete

2. `src/timefm_split.py`
   - keep

3. `src/timefm_targets.py`
   - keep

4. `src/timefm_features.py`
   - keep

### Replace in-place

1. `src/timefm_forecaster.py`
   Replace the v1 runtime wrapper with a TimesFM 2.5 Transformers inference wrapper.

   New responsibilities:
   - define `TimeFMModelHandle` for TimesFM 2.5 model + tokenizer/config metadata if needed
   - load model through Transformers
   - expose `forecast_sample()` and `forecast_batch()` on `TimeFMSample`
   - normalize model outputs into the existing `TimeFMForecastResult`
   - optionally preserve a fake backend for tests only

   Remove:
   - `TIMEFM_V1_DEFAULTS`
   - direct `timesfm.TimesFm` construction
   - `load_from_checkpoint(repo_id=...)`
   - v1 quantile extraction assumptions

2. `src/timefm_loader_spike.py`
   Replace defaults and report structure to exercise TimesFM 2.5.

   New defaults:
   - backend name should reflect TimesFM 2.5 transformers, e.g. `timesfm-2.5-transformers`
   - model id should default to `google/timesfm-2.5-200m-transformers`

   New responsibility:
   - prove one-sample inference works with the new wrapper
   - optionally report device and dtype in the spike output

3. `src/run_timefm_forecasts.py`
   Keep the overall artifact flow, but retarget defaults and handle 2.5 wrapper semantics.

   Additions likely needed:
   - batch-size control
   - device option (`cpu`, `mps`, `cuda`, `auto`)
   - maybe `max_context_len` or model-config-aware clipping

4. `pyproject.toml`
   Replace v1 dependency set.

   Remove:
   - `timesfm = "^1.0.0"`

   Add:
   - `transformers`
   - `peft`
   - `accelerate`
   - keep `torch`

### Split into new files

1. Create `src/timefm25_torch_dataset.py`
   Purpose:
   - convert existing `TimeFMSample`/weekly series into train/eval datasets for TimesFM 2.5 fine-tuning
   - support random window sampling and last-window validation similar to Google's example

2. Create `src/timefm25_finetune.py`
   Purpose:
   - LoRA fine-tuning entry point using Transformers + PEFT
   - consume chronological split data from existing weekly files

3. Create `src/timefm25_evaluate.py`
   Purpose:
   - compare base vs LoRA-adapted model on validation/test splits
   - emit MAE and directional metrics

### Tests to replace or split

1. `test/test_timefm_phase2.py`
   Split its responsibilities.

   Keep / adapt:
   - fake backend structured result test
   - loader spike report shape test
   - JSONL artifact writing test

   Remove / replace:
   - v1 constructor/load-from-checkpoint test
   - v1 `.forecast([context], freq=[0])` assumptions

2. Create `test/test_timefm25_forecaster.py`
   Add tests for:
   - loading the 2.5 wrapper via stubbed Transformers model class
   - inference output normalization into `TimeFMForecastResult`
   - handling horizon clipping / shape normalization

3. Create `test/test_timefm25_torch_dataset.py`
   Add tests for:
   - random training window extraction
   - last-window validation extraction
   - no-leakage and exact shape guarantees
   - full-context requirement without zero-padding

4. Create `test/test_timefm25_finetune_config.py`
   Add tests for:
   - LoRA config construction
   - split-aware dataset loading
   - output directory/checkpoint path logic

---

## Recommended target API for the new forecaster

### Inference wrapper contract

Keep these public functions stable if possible:
- `load_model(...)`
- `forecast_sample(...)`
- `forecast_batch(...)`

Recommended new defaults:
- `model_id='google/timesfm-2.5-200m-transformers'`
- `backend_name='timesfm-2.5-transformers'`

Recommended `load_model(...)` inputs:
- `model_id: str`
- `backend_name: str = 'timesfm-2.5-transformers'`
- `device: str = 'auto'`
- `torch_dtype: str | None = None`

Recommended `forecast_sample(...)` behavior:
- input: existing `TimeFMSample`
- call the model in inference mode with `past_values=context_tensor`
- use model outputs to populate:
  - `predicted_mean`
  - optional lower/upper intervals if exposed
- preserve repo metadata:
  - `symbol`
  - `source_file`
  - `context_window`
  - `forecast_horizon`
  - forecast timestamps

### Fine-tuning contract

Create a separate training path rather than overloading `timefm_forecaster.py`.

Recommended public functions in new files:
- `build_random_window_datasets(...)`
- `train_lora_adapter(...)`
- `evaluate_base_and_adapter(...)`

---

## TDD implementation order

### Task 1: lock the migration boundary in tests

Objective: write failing tests that describe the new 2.5 direction before changing runtime code.

Files:
- Modify: `test/test_timefm_phase2.py`
- Create: `test/test_timefm25_forecaster.py`

Write failing tests for:
1. `load_model()` default model id is `google/timesfm-2.5-200m-transformers`
2. `load_model()` no longer imports v1 `timesfm.TimesFm`
3. a stubbed Transformers 2.5 model can produce a `TimeFMForecastResult`
4. existing JSONL forecast artifact shape remains stable

Verification:
- `./.venv/bin/python -m unittest discover -s test -p 'test_timefm25*.py'`
- expected first run: FAIL

### Task 2: migrate dependency declarations

Objective: change the repo dependency plan to the new runtime.

Files:
- Modify: `pyproject.toml`

Changes:
- remove `timesfm = "^1.0.0"`
- add `transformers`, `peft`, `accelerate`
- keep `torch`

Verification:
- read back `pyproject.toml`
- later install command succeeds in a compatible environment

### Task 3: replace the inference wrapper

Objective: make `src/timefm_forecaster.py` target TimesFM 2.5 Transformers.

Files:
- Modify: `src/timefm_forecaster.py`
- Test: `test/test_timefm25_forecaster.py`

Steps:
1. delete v1 defaults/constants
2. keep or rename fake backend to remain test-only
3. add new 2.5 `load_model()` path
4. normalize model outputs into `TimeFMForecastResult`
5. keep `forecast_batch()` API stable

Verification:
- targeted test passes
- legacy artifact-writing test still passes

### Task 4: retarget the spike and artifact runner

Objective: keep CLI flows working with the new backend.

Files:
- Modify: `src/timefm_loader_spike.py`
- Modify: `src/run_timefm_forecasts.py`
- Test: relevant spike/runner tests

Steps:
1. change defaults to 2.5 model/backend
2. thread through `device` and optional batch controls
3. ensure spike report exposes model/backend/device
4. ensure JSONL writer still emits the same record keys

Verification:
- targeted tests pass
- a real spike can be attempted in the final runtime

### Task 5: add fine-tuning dataset helpers

Objective: build a clean bridge from current weekly data to LoRA training windows.

Files:
- Create: `src/timefm25_torch_dataset.py`
- Create: `test/test_timefm25_torch_dataset.py`

Steps:
1. implement random-window train dataset
2. implement last-window validation dataset
3. require full context + horizon coverage
4. avoid zero-padding and external normalization

Verification:
- dataset tests pass

### Task 6: add LoRA fine-tuning entry point

Objective: create a real project-native fine-tuning script.

Files:
- Create: `src/timefm25_finetune.py`
- Create: `test/test_timefm25_finetune_config.py`

Steps:
1. load base TimesFM 2.5 transformers checkpoint
2. attach LoRA via PEFT
3. train on manifest-defined train split
4. validate on manifest-defined val split
5. save adapter to output directory

Verification:
- config/unit tests pass
- a reduced dry-run can be attempted once deps are installed

---

## Exact files to touch first

Immediate first-wave files:
- `test/test_timefm_phase2.py`
- `test/test_timefm25_forecaster.py` (new)
- `src/timefm_forecaster.py`
- `src/timefm_loader_spike.py`
- `src/run_timefm_forecasts.py`
- `pyproject.toml`

Second-wave files:
- `src/timefm25_torch_dataset.py` (new)
- `test/test_timefm25_torch_dataset.py` (new)
- `src/timefm25_finetune.py` (new)
- `test/test_timefm25_finetune_config.py` (new)

---

## Runtime notes for this machine

Verified current blocker state:
- project `.venv` currently does not have `torch`, `transformers`, `peft`, or `accelerate`
- before any real spike can run, those dependencies must be installed

That means the next practical step is:
1. write the failing 2.5 tests first
2. update dependency declarations
3. install the new stack in a compatible env
4. then run the real 2.5 loader spike

---

## Success criteria

The migration is successful when all of the following are true:
- repo no longer depends on `timesfm==1.0.0` as the primary path
- `src/timefm_forecaster.py` loads TimesFM 2.5 through a Transformers-based path
- `src/timefm_loader_spike.py` runs a 2.5 one-sample forecast
- `src/run_timefm_forecasts.py` writes JSONL forecast artifacts with the new backend
- a separate LoRA fine-tuning script exists for chronological train/val splits
- test suite reflects 2.5 semantics instead of v1 semantics
