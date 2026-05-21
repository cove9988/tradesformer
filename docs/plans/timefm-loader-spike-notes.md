# TimeFM loader spike notes

Goal
- verify the repo can construct TimeFM-style forecast samples and run one forecast through a pluggable backend interface before wiring heavier dependencies

Current implementation
- `src/timefm_loader_spike.py` builds one sample from a weekly CSV and runs it through `src.timefm_forecaster.py`
- default backend is `fake-timesfm` so Phase 2 is testable without network installs or heavyweight runtime assumptions
- `load_model(..., backend_name='timesfm')` is reserved for real package integration after package/runtime validation

Observed package state in this environment
- `timesfm` is not currently installed in the project venv
- `pip index versions timesfm` reports version `1.0.0`

Recommended next real-package validation
1. add `timesfm` to `pyproject.toml`
2. run `poetry install`
3. execute a real spike against one weekly CSV using `--backend-name timesfm`
4. document actual constructor/loading API once confirmed from the installed package

Apple Silicon / CPU / GPU caveats
- no verified Apple Silicon runtime path has been confirmed yet in this repo
- until a real `timesfm` package load succeeds locally, treat `timesfm` backend support as provisional
- keep the fake backend path available for unit tests and pipeline development so the project is not blocked by runtime packaging issues

Why this staged approach
- it preserves TDD and lets the repo gain a stable forecaster/artifact pipeline first
- it isolates uncertain third-party package details behind one module instead of spreading experimental code across training/eval scripts
