# MT5 + TimesFM 2.5 Forecasting Project

Forecast DJ30, NAS100, XAUUSD, WTI, EURUSD (and more) using
Google's TimesFM 2.5 fine-tuned with LoRA/DoRA on your own data,
with full covariate support (moon phases, economic calendar, sessions, etc.).

## Setup

```powershell
# From the repo root (timesfm/)
uv pip install -e ".[torch,xreg]"
pip install ephem pandas-market-calendars MetaTrader5
```

## Workflow

1. **Generate static data** (one-time):
   ```
   python mt5/scripts/generate_moon_phases.py
   python mt5/scripts/generate_economic_calendar.py
   ```

2. **Export MT5 data** (run while MT5 terminal is open):
   ```
   python mt5/scripts/export_mt5_data.py
   ```

3. **Build covariate dataset** (merges everything):
   ```
   python mt5/scripts/build_covariates.py
   ```

4. **Fine-tune** with LoRA:
   ```
   python mt5/scripts/finetune.py
   ```

5. **Forecast**:
   ```
   python mt5/scripts/forecast.py
   ```

## Instruments

Configured in `mt5/config.py`. Default:
- DJ30, NAS100, XAUUSD.i, WTI, EURUSD.i

## Data

- `mt5/data/moon_phases/` — Moon phase data (2000–2026)
- `mt5/data/economic_calendar/` — Major economic events (2000–2026)
- `mt5/data/mt5_exports/` — Raw OHLCV from MetaTrader 5
- `mt5/data/covariates/` — Assembled covariate CSVs ready for training
- `mt5/checkpoints/` — Fine-tuned adapter weights
