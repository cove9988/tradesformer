"""MT5 + TimesFM project configuration."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
MOON_DIR = DATA_DIR / "moon_phases"
ECON_DIR = DATA_DIR / "economic_calendar"
MT5_DIR = DATA_DIR / "mt5_exports"
COV_DIR = DATA_DIR / "covariates"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

for d in [MOON_DIR, ECON_DIR, MT5_DIR, COV_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Instruments ──────────────────────────────────────────────────────────
# Keys: display name used in filenames / columns
# Values: dict with MT5 symbol, asset class, base currency, timeframe
INSTRUMENTS = {
    "DJ30": {
        "mt5_symbol": "DJ30",
        "asset_class": "index",
        "base_currency": "USD",
    },
    "NAS100": {
        "mt5_symbol": "NAS100",
        "asset_class": "index",
        "base_currency": "USD",
    },
    "XAUUSD": {
        "mt5_symbol": "XAUUSD.i",
        "asset_class": "commodity",
        "base_currency": "USD",
    },
    "WTI": {
        "mt5_symbol": "WTI",
        "asset_class": "commodity",
        "base_currency": "USD",
    },
    "EURUSD": {
        "mt5_symbol": "EURUSD.i",
        "asset_class": "forex",
        "base_currency": "EUR",
    },
    "USDJPY": {
        "mt5_symbol": "USDJPY.i",
        "asset_class": "forex",
        "base_currency": "USD",
    },
    "GBPUSD": {
        "mt5_symbol": "GBPUSD.i",
        "asset_class": "forex",
        "base_currency": "GBP",
    },
    "AUDUSD": {
        "mt5_symbol": "AUDUSD.i",
        "asset_class": "forex",
        "base_currency": "AUD",
    },
    "USDCAD": {
        "mt5_symbol": "USDCAD.i",
        "asset_class": "forex",
        "base_currency": "USD",
    },
    "USDCHF": {
        "mt5_symbol": "USDCHF.i",
        "asset_class": "forex",
        "base_currency": "USD",
    },
    "NZDUSD": {
        "mt5_symbol": "NZDUSD.i",
        "asset_class": "forex",
        "base_currency": "NZD",
    },
    "EURGBP": {
        "mt5_symbol": "EURGBP.i",
        "asset_class": "forex",
        "base_currency": "EUR",
    },
    "EURJPY": {
        "mt5_symbol": "EURJPY.i",
        "asset_class": "forex",
        "base_currency": "EUR",
    },
    "GBPJPY": {
        "mt5_symbol": "GBPJPY.i",
        "asset_class": "forex",
        "base_currency": "GBP",
    },
}

# ── Timeframes ───────────────────────────────────────────────────────────
# MT5 timeframe constant name → (label, bars per day approx)
TIMEFRAME = "TIMEFRAME_H1"  # Change to TIMEFRAME_M15, TIMEFRAME_D1, etc.
BARS_TO_PULL = 50_000  # Max history to pull per instrument

# ── Date range for static data (moon, econ calendar) ────────────────────
DATA_START = "2000-01-01"
DATA_END = "2026-04-08"

# ── Training defaults ─────────────────────────────────────────────────────
# Tuned for capital preservation on live trading. Rationale per parameter:
#
#   CONTEXT_LEN = 512   Shorter context = more training windows from the same
#                        data, reducing overfit. 512 H1 bars ≈ 21 trading days
#                        — plenty of regime information for intraday horizons.
#
#   HORIZON_LEN = 24    24 × H1 = 1 trading day lookahead. Forecasting further
#                        out in financial markets is exponentially harder.
#
#   BATCH_SIZE = 128     Larger batches → more stable gradient estimates.
#                        The RTX 5090 can handle it with bf16.
#
#   NUM_EPOCHS = 100     Max cap. Early stopping (patience 15) will halt well
#                        before this in practice.
#
#   ADAPTER_TYPE = lora  LoRA keeps the number of trainable params tiny
#                        (∼100K), which is the single most important defence
#                        against overfitting on non-stationary financial data.
#
#   LORA_RANK = 4        The minimum effective rank. With 14 instruments and
#                        ~638 K bars the data-to-param ratio must stay high.
#                        rank 4 → ~102 K trainable params across 2 layers.
#
#   LORA_ALPHA = 8.0     alpha / rank = 2.0 is the standard LoRA scaling.
#
#   LORA_DROPOUT = 0.15  Moderate dropout on the adapter path. Prevents the
#                        LoRA matrices from memorising specific price patterns.
#
#   NUM_ADAPTER_LAYERS = 2   Only adapt the last 2 of 20 transformer layers
#                             (10%). The pretrained representations in the
#                             earlier layers generalise better than anything
#                             we can teach on our small dataset.
#
#   LEARNING_RATE = 5e-5      Conservative LR. Financial loss surfaces are
#                              noisy — large steps create unstable adapters.
#
#   WEIGHT_DECAY = 0.1        Strong L2 regularisation. Penalises large LoRA
#                              weight magnitudes, keeping the adapter close to
#                              the pretrained model.
#
#   WARMUP_RATIO = 0.15       15 % of training steps at linearly increasing LR.
#                              Avoids early gradient explosions from the random
#                              LoRA initialisation.
#
#   EARLY_STOPPING_PATIENCE = 15    Wait 15 epochs of no val improvement
#                                    before stopping. Financial val loss is
#                                    noisy — short patience stops too early.
#
CONTEXT_LEN = 512
HORIZON_LEN = 48
BATCH_SIZE = 128
NUM_EPOCHS = 100
ADAPTER_TYPE = "lora"
LORA_RANK = 4
LORA_ALPHA = 8.0
LORA_DROPOUT = 0.15
NUM_ADAPTER_LAYERS = 2
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.15
EARLY_STOPPING_PATIENCE = 15

# ── Forex session times (UTC hours) ─────────────────────────────────────
SESSIONS = {
    "tokyo":  (0, 9),
    "london": (7, 16),
    "ny":     (12, 21),
}

# ── Major economic events for the calendar ───────────────────────────────
# (event_name, typical_day_description, impact_level)
MAJOR_EVENTS = [
    ("FOMC_Decision", "Federal Reserve interest rate decision", "high"),
    ("NFP", "Non-Farm Payrolls (first Friday of month)", "high"),
    ("CPI_US", "US Consumer Price Index", "high"),
    ("PPI_US", "US Producer Price Index", "medium"),
    ("GDP_US", "US Gross Domestic Product", "high"),
    ("Retail_Sales_US", "US Retail Sales", "medium"),
    ("ISM_Manufacturing", "ISM Manufacturing PMI", "medium"),
    ("ISM_Services", "ISM Services PMI", "medium"),
    ("Unemployment_Claims", "Weekly Initial Jobless Claims", "medium"),
    ("ECB_Decision", "ECB interest rate decision", "high"),
    ("BOE_Decision", "Bank of England rate decision", "medium"),
    ("BOJ_Decision", "Bank of Japan rate decision", "medium"),
    ("CPI_EU", "Eurozone HICP/CPI", "medium"),
    ("GDP_EU", "Eurozone GDP", "medium"),
]
