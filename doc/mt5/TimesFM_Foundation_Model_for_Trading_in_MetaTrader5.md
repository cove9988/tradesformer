# Foundation Models in Trading: Time Series Forecasting with Google's TimesFM 2.5 in MetaTrader 5

## Introduction

Time series forecasting is one of the oldest yet most enduring challenges in quantitative trading. Traditional approaches — from ARIMA to exponential smoothing — rely heavily on manual feature engineering, careful hyperparameter selection, and the assumption that the statistical properties of the series remain stable over time. Deep learning models such as DeepAR and PatchTST improved upon these classical methods, but they still demand long training cycles on domain-specific data before they can produce useful results.

In the natural language processing domain, we have witnessed a paradigm shift: large foundation models, pretrained on vast corpora and then adapted to specific tasks, have displaced the "train from scratch" workflow. The question naturally arises: _can this transfer learning revolution work for time series?_

Google Research answered this affirmatively with **TimesFM** (Time Series Foundation Model), introduced in the paper ["A decoder-only foundation model for time-series forecasting"](https://arxiv.org/abs/2310.10688) and accepted at ICML 2024. TimesFM is a 200-million-parameter decoder-only transformer pretrained on a corpus of 100 billion real-world time points. Despite being orders of magnitude smaller than contemporary LLMs, it achieves remarkable zero-shot performance across domains and granularities — matching or outperforming supervised models that were explicitly trained on the target datasets.

What makes this particularly interesting for algorithmic traders is the ability to _fine-tune_ this pretrained knowledge using our own financial data. By applying Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters, we can specialize TimesFM on our specific instruments while keeping the number of trainable parameters below 100K — a critical safeguard against overfitting on non-stationary market data.

In this article, we build a complete end-to-end pipeline that:

1. Exports OHLCV data from MetaTrader 5 for 14 instruments (forex pairs, indices, commodities)
2. Constructs a rich covariate dataset incorporating moon phases, economic calendar events, market sessions, and technical features
3. Fine-tunes TimesFM 2.5 with LoRA adapters on this financial data
4. Generates probabilistic forecasts with quantile estimates (10th, 50th, 90th percentiles)
5. Exports the forecasts back into MetaTrader 5 as CSV files
6. Visualizes the predictions directly on the chart via a custom MQL5 indicator with confidence bands

The full Python codebase (`mt5/` folder) is attached to this article as a ZIP archive. The underlying TimesFM library must be cloned from the official [GitHub repository](https://github.com/google-research/timesfm).


## 1. TimesFM Architecture: The Concept

Before diving into implementation details, let us briefly examine the key ideas behind TimesFM that make it suitable for financial time series.

### 1.1 Decoder-only Transformer for Time Series

LLMs are typically trained in a decoder-only fashion: text is tokenized, tokens are fed through stacked causal transformer layers, and each output position predicts the next token. TimesFM applies the same principle to time series, but with a critical adaptation — instead of text tokens, the model operates on **patches**: contiguous groups of time points.

Given an input patch length of 32 and an output patch length of 128, the model simultaneously learns to:
- Use the first 32 points to forecast the next 128
- Use the first 64 points to forecast points 65–192
- Use the first 96 points to forecast points 97–224
- And so on...

This training strategy has a concrete advantage during inference: if we need to forecast 256 points into the future from a context of 256 points, the model achieves this in just 2 generation steps rather than 8, dramatically reducing error accumulation.

```
Input Series (512 points)
  ┌────────────────────────────────────────────┐
  │ patch_1 │ patch_2 │ ... │ patch_16         │
  └────────────────────────────────────────────┘
        │           │               │
        ▼           ▼               ▼
  ┌─────────────────────────────────────┐
  │    Stacked Causal Transformer       │
  │    (Self-Attention + FFN layers)    │
  └─────────────────────────────────────┘
        │           │               │
        ▼           ▼               ▼
  ┌────────────────────────────────────────────┐
  │ forecast_1 │ forecast_2 │ ... │ forecast_N │
  └────────────────────────────────────────────┘
```

### 1.2 Pretraining Data

TimesFM 2.5 is pretrained on:

- **GiftEvalPretrain** — a curated time series collection from Salesforce
- **Wikimedia Pageviews** — web traffic data mirroring real-world trends and seasonality
- **Google Trends** — search interest data capturing macroeconomic and cultural cycles
- **Synthetic and augmented data** — teaching the model the "grammar" of temporal patterns

This diverse pretraining means the model already understands concepts like trends, seasonality, level shifts, and volatility clustering before ever seeing a single financial candlestick.

### 1.3 Why Fine-Tune Instead of Train from Scratch?

Financial time series present a unique challenge: the amount of usable data is limited (markets are open roughly 252 days/year), and the data is inherently non-stationary. Training a deep model from scratch on our 14 instruments would virtually guarantee overfitting.

LoRA (Low-Rank Adaptation) solves this elegantly. Instead of updating all 200M parameters, we inject small trainable low-rank matrices into only the last 2 of 20 transformer layers. This yields approximately 102K trainable parameters — a data-to-parameter ratio that remains healthy even with our relatively small dataset. The pretrained representations in the earlier 18 layers generalize far better than anything we could learn from our limited financial data.

### 1.4 Covariates: Beyond Raw Prices

A key feature of TimesFM 2.5 is its support for external covariates through the `xreg` mechanism. Our pipeline exploits this by feeding the model:

| Covariate Group | Features | Rationale |
|---|---|---|
| **Time cyclical** | hour_sin/cos, dow_sin/cos, woy_sin/cos, moy_sin/cos, dom_sin/cos | Markets exhibit strong intraday and weekly seasonality |
| **Market sessions** | session_tokyo, session_london, session_ny, session_overlap | Volatility and liquidity regimes change with trading sessions |
| **Moon phases** | phase_sin/cos, is_new_moon, is_full_moon, days_to_new, days_to_full | Empirically studied correlation with market sentiment |
| **Economic calendar** | econ_FOMC, econ_NFP, econ_CPI_US, econ_GDP_US, days_to_event | Major macro events drive regime changes |
| **Technical proxies** | atr_norm, volume_norm, spread_norm, returns_1h/4h/24h | Recent volatility and momentum context |

All time-based features use sin/cos cyclical encoding to avoid discontinuities at period boundaries — a practice well-established in time series modeling.


## 2. Project Architecture

The pipeline is organized as a set of modular Python scripts within the `mt5/` folder, orchestrated by a single entry point:

```
mt5/
├── config.py                         # All paths, instruments, hyperparameters
├── run_pipeline.py                   # Orchestrator: runs steps 1–7 in sequence
├── .env                              # MT5 credentials + FRED API key
├── scripts/
│   ├── export_mt5_data.py           # Step 1: Pull OHLCV from MetaTrader 5
│   ├── generate_moon_phases.py      # Step 2: Astronomical moon calculations
│   ├── generate_economic_calendar.py # Step 3: FRED API + FOMC dates
│   ├── build_covariates.py          # Step 4: Merge all features
│   ├── finetune.py                  # Step 5: LoRA/DoRA fine-tuning
│   ├── forecast.py                  # Step 6: Generate probabilistic forecasts
│   ├── export_to_mt5.py            # Step 7: Write CSVs to MQL5/Files/
│   └── visualize_forecast.py       # Optional: HTML forecast dashboard
├── mql5/
│   └── TimesFM_Forecast.mq5        # Chart indicator for MetaTrader 5
├── checkpoints/                     # Saved LoRA adapter weights
└── data/
    ├── mt5_exports/                 # Raw OHLCV CSVs
    ├── moon_phases/                 # Precomputed lunar data
    ├── economic_calendar/           # FRED + FOMC event data
    ├── covariates/                  # Final merged datasets
    └── forecast_results.json        # Full forecast output
```

Each script is self-contained and can be run independently, but the `run_pipeline.py` orchestrator ties them together:

```
python mt5/run_pipeline.py                  # Full pipeline (steps 1–7)
python mt5/run_pipeline.py --skip-finetune  # Reuse existing adapter
python mt5/run_pipeline.py --from-step 4    # Resume from covariates
python mt5/run_pipeline.py --only 6 7       # Forecast + export only
```


## 3. Installation and Setup

### 3.1 Clone the TimesFM Repository

TimesFM is not yet available via `pip install` from PyPI for the 2.5 version. Clone the official repository:

```bash
git clone https://github.com/google-research/timesfm.git
cd timesfm
```

### 3.2 Install Dependencies

The recommended approach uses `uv` for fast dependency resolution:

```bash
# Install TimesFM in editable mode with torch and covariate support
uv pip install -e ".[torch,xreg]"

# Additional dependencies for the MT5 pipeline
pip install ephem MetaTrader5 python-dotenv plotly fredapi
```

For systems without `uv`:

```bash
pip install -e ".[torch,xreg]"
pip install ephem MetaTrader5 python-dotenv plotly fredapi
```

**Hardware note:** Fine-tuning runs on any CUDA GPU. The script automatically detects NVIDIA GPUs, Intel XPU/NPU, and falls back to CPU. For the default configuration (batch_size=128, context=512, bf16), an NVIDIA GPU with 8+ GB VRAM is recommended. CPU training works but is significantly slower.

### 3.3 Extract the mt5 Folder

Download the `mt5.zip` attached to this article and extract it into the repository root so that the `mt5/` folder sits alongside `src/`, `peft/`, and `tests/`:

```
timesfm/
├── mt5/            ← extracted from the attachment
├── peft/
├── src/
├── tests/
├── pyproject.toml
└── ...
```

### 3.4 Configure Credentials

Create (or edit) the file `mt5/.env` with your MetaTrader 5 login credentials and a FRED API key:

```ini
# MetaTrader 5 account
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Server

# Federal Reserve FRED API (free: https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_API_KEY=your_fred_api_key
```

The MT5 credentials are required for Steps 1 and 7 (data export and forecast export). The FRED API key is needed only for Step 3 (economic calendar generation).

### 3.5 Configure Instruments

All instruments are defined in `mt5/config.py`. The default configuration covers 14 symbols across three asset classes:

```python
INSTRUMENTS = {
    "DJ30":    {"mt5_symbol": "DJ30",      "asset_class": "index",     "base_currency": "USD"},
    "NAS100":  {"mt5_symbol": "NAS100",    "asset_class": "index",     "base_currency": "USD"},
    "XAUUSD":  {"mt5_symbol": "XAUUSD.i",  "asset_class": "commodity", "base_currency": "USD"},
    "WTI":     {"mt5_symbol": "WTI",       "asset_class": "commodity", "base_currency": "USD"},
    "EURUSD":  {"mt5_symbol": "EURUSD.i",  "asset_class": "forex",    "base_currency": "EUR"},
    "USDJPY":  {"mt5_symbol": "USDJPY.i",  "asset_class": "forex",    "base_currency": "USD"},
    "GBPUSD":  {"mt5_symbol": "GBPUSD.i",  "asset_class": "forex",    "base_currency": "GBP"},
    "AUDUSD":  {"mt5_symbol": "AUDUSD.i",  "asset_class": "forex",    "base_currency": "AUD"},
    "USDCAD":  {"mt5_symbol": "USDCAD.i",  "asset_class": "forex",    "base_currency": "USD"},
    "USDCHF":  {"mt5_symbol": "USDCHF.i",  "asset_class": "forex",    "base_currency": "USD"},
    "NZDUSD":  {"mt5_symbol": "NZDUSD.i",  "asset_class": "forex",    "base_currency": "NZD"},
    "EURGBP":  {"mt5_symbol": "EURGBP.i",  "asset_class": "forex",    "base_currency": "EUR"},
    "EURJPY":  {"mt5_symbol": "EURJPY.i",  "asset_class": "forex",    "base_currency": "EUR"},
    "GBPJPY":  {"mt5_symbol": "GBPJPY.i",  "asset_class": "forex",    "base_currency": "GBP"},
}
```

Adjust the `mt5_symbol` values to match your broker's naming convention. Some brokers use suffixes like `.i`, `.pro`, or `m` — check the Market Watch panel in your MetaTrader 5 terminal.


## 4. Implementation: Step by Step

### 4.1 Step 1 — Export OHLCV Data from MetaTrader 5

The `export_mt5_data.py` script connects to a running MetaTrader 5 terminal via the `MetaTrader5` Python package and pulls historical OHLCV data for all configured instruments.

```python
def export_instrument(name: str, info: dict) -> pd.DataFrame | None:
    symbol = info["mt5_symbol"]

    if not mt5.symbol_select(symbol, True):
        print(f"  WARNING: Symbol {symbol} not available in MT5. Skipping.")
        return None

    tf_const = getattr(mt5, TIMEFRAME, None)
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, BARS_TO_PULL)

    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df = df[["datetime", "open", "high", "low", "close", "tick_volume", "spread"]]
    return df.sort_values("datetime").reset_index(drop=True)
```

The `BARS_TO_PULL` parameter (default: 50,000) controls how much history to retrieve per symbol. For H1 timeframe, 50,000 bars covers roughly 8 years of market data — sufficient for fine-tuning.

The script writes one CSV per instrument to `mt5/data/mt5_exports/`:

```
EURUSD_H1.csv
XAUUSD_H1.csv
DJ30_H1.csv
...
```

**Important:** The MetaTrader 5 terminal must be running and logged in before executing this step.

```bash
python mt5/scripts/export_mt5_data.py
```

### 4.2 Steps 2–3 — Generate Static Feature Datasets

Two scripts precompute datasets that are independent of the market data and need to be generated only once.

#### Moon Phases

The `generate_moon_phases.py` script uses the `ephem` (PyEphem) astronomical library to compute precise moon phase data at hourly resolution from 2000 to 2026:

```python
def build_hourly_moon_data(start: str, end: str) -> pd.DataFrame:
    dt_range = pd.date_range(start=start, end=end, freq="h")
    records = []

    for dt in dt_range:
        m = ephem.Moon()
        m.compute(dt.strftime("%Y/%m/%d %H:%M:%S"))
        phase_pct = float(m.phase) / 100.0  # 0=new, ~1=full
        records.append({"datetime": dt, "phase_pct": phase_pct})

    df = pd.DataFrame(records)
    df["phase_sin"] = np.sin(2 * np.pi * df["phase_pct"])
    df["phase_cos"] = np.cos(2 * np.pi * df["phase_pct"])
    # ... binary flags for new/full/quarter + days_to_next
    return df
```

The lunar cycle has been a subject of empirical research in financial markets. Whether the effect is genuine or coincidental, including it as a covariate lets the model decide its relevance during training.

#### Economic Calendar

The `generate_economic_calendar.py` script fetches actual release dates for major US economic indicators from the Federal Reserve Economic Data (FRED) REST API:

- **CPI** (Consumer Price Index) — high impact
- **PPI** (Producer Price Index) — medium impact
- **GDP** (Gross Domestic Product) — high impact
- **NFP** (Nonfarm Payrolls) — high impact
- **Retail Sales** — medium impact
- **FOMC** meetings — sourced from the Federal Reserve website (all dates 2000–2026 hardcoded from the official calendar)

Each event becomes a binary feature at hourly resolution, plus a `days_to_event` feature that creates a gradient of anticipation as events approach.

```bash
python mt5/scripts/generate_moon_phases.py
python mt5/scripts/generate_economic_calendar.py
```

### 4.3 Step 4 — Build Merged Covariates

The `build_covariates.py` script merges everything into a single DataFrame per instrument:

```
OHLCV data  ─┐
Moon phases  ─┤
Econ calendar ┼──→ merge on datetime ──→ add time features ──→ EURUSD_H1_covariates.csv
Sessions     ─┤
Technical    ─┘
```

The merge is performed on hourly-rounded timestamps. Time features are computed dynamically:

```python
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["datetime"]

    # Cyclical encoding avoids discontinuities
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24).astype(np.float32)

    df["dow_sin"]  = np.sin(2 * np.pi * dt.dt.dayofweek / 7).astype(np.float32)
    df["dow_cos"]  = np.cos(2 * np.pi * dt.dt.dayofweek / 7).astype(np.float32)

    # ... week of year, month of year, day of month
    return df
```

Market sessions are encoded as binary flags based on UTC hour ranges defined in `config.py`. The London/NY overlap session — known for the highest forex liquidity and volatility — gets its own flag.

Technical proxies (ATR, volume, spread) are normalized using a 24-hour rolling mean to make them scale-invariant across different instruments.

```bash
python mt5/scripts/build_covariates.py
```

The output is one comprehensive CSV per instrument in `mt5/data/covariates/`, each containing 40+ feature columns.

### 4.4 Step 5 — Fine-Tune TimesFM with LoRA

This is the core of the pipeline. The `finetune.py` script loads the pretrained TimesFM 2.5 model from HuggingFace and applies LoRA fine-tuning using the PEFT framework included in the repository.

#### Hardware Detection

The script begins with automatic hardware detection, identifying CUDA GPUs, Intel XPU (Arc), and NPU devices:

```python
def detect_hardware():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            print(f"  CUDA:{i}  {props.name}  ({vram_gb:.1f} GB VRAM)")
    # ... Intel XPU, NPU detection ...
    # Select best GPU or fall back to CPU
```

#### Hyperparameter Rationale

Every training hyperparameter in `config.py` is deliberately tuned for capital preservation on live trading. This warrants detailed explanation because the choices differ significantly from typical deep learning wisdom:

```python
CONTEXT_LEN         = 512     # 512 H1 bars ≈ 21 trading days
HORIZON_LEN         = 48      # 48 × H1 = 2 trading days lookahead
BATCH_SIZE          = 128     # Larger batches → stable gradient estimates
NUM_EPOCHS          = 100     # Max cap — early stopping halts much earlier
ADAPTER_TYPE        = "lora"  # Keeps trainable params ~100K
LORA_RANK           = 4       # Minimum effective rank
LORA_ALPHA          = 8.0     # alpha/rank = 2.0 (standard LoRA scaling)
LORA_DROPOUT        = 0.15    # Prevents memorizing price patterns
NUM_ADAPTER_LAYERS  = 2       # Only last 2 of 20 transformer layers
LEARNING_RATE       = 5e-5    # Conservative — financial loss surfaces are noisy
WEIGHT_DECAY        = 0.1     # Strong L2 regularization
WARMUP_RATIO        = 0.15    # 15% linear warmup to avoid gradient explosions
EARLY_STOPPING_PATIENCE = 15  # Financial val loss is noisy — patience is needed
```

**`LORA_RANK = 4`** deserves special attention. With 14 instruments and approximately 638K bars of data, the data-to-parameter ratio must remain high. Rank 4 produces ~102K trainable parameters across 2 layers — enough to learn domain-specific adjustments without enough capacity to memorize noise.

**`NUM_ADAPTER_LAYERS = 2`** means we only adapt the final 10% of the transformer stack. The rationale is that earlier layers encode general temporal patterns (trends, seasonality, level shifts) that transfer well across domains. Only the later layers need adjustment for financial-specific pattern recognition.

#### Training Loop

The model is loaded and adapted:

```python
model_wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch",
    torch_compile=False,  # Must disable for fine-tuning
)

config = PEFTConfig(
    adapter_type="lora",
    lora_rank=4,
    lora_alpha=8.0,
    lora_dropout=0.15,
    num_adapter_layers=2,
    # ... other parameters
)

trainer = PEFTTrainer(model_wrapper.model, config)
history = trainer.fit(train_ds, val_ds)
```

Data is split chronologically (80/20) to prevent look-ahead bias — a critical detail for financial data. Training windows use 50% overlap for diversity:

```python
for s in all_series:
    split_idx = int(len(s) * 0.8)
    train_series.append(s[:split_idx])
    val_series.append(s[split_idx:])
```

The adapter weights are saved as safetensors files in `mt5/checkpoints/`. The best checkpoint (by validation loss) is saved as `best_adapter.safetensors`, and per-epoch checkpoints are also retained.

```bash
python mt5/scripts/finetune.py
# Or with custom parameters:
python mt5/scripts/finetune.py --adapter_type dora --num_epochs 30 --lora_rank 8
```

### 4.5 Step 6 — Generate Forecasts

The `forecast.py` script loads the fine-tuned model and generates probabilistic forecasts for all instruments:

```python
# Load base model
model_wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch", torch_compile=True
)

# Inject LoRA adapter and load fine-tuned weights
peft_config = PEFTConfig(lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA, ...)
inject_adapters(model_wrapper.model, peft_config)
load_adapter_weights(model_wrapper.model, adapter_path)
```

For each instrument, the script:

1. Loads the last `CONTEXT_LEN` bars of close prices from the covariate CSV
2. Constructs future covariates for the forecast horizon (time features are computed exactly; technical features are forward-filled; economic events are zeroed for unknown future)
3. Calls the model's `forecast_with_covariates` method

```python
point, quantiles = model.forecast_with_covariates(
    inputs=[close_context],
    dynamic_numerical_covariates=dynamic_numerical,
    static_categorical_covariates={
        "asset_class": [asset_class],
        "instrument": [name],
    },
    xreg_mode="xreg + timesfm",
)
```

The `xreg_mode="xreg + timesfm"` setting means the model blends its internal time series representation with the external covariate information — giving it the best of both worlds.

The output is saved to `mt5/data/forecast_results.json`, containing for each instrument:
- Point forecast (mean)
- 10th percentile (q10) — lower confidence bound
- 50th percentile (q50) — median
- 90th percentile (q90) — upper confidence bound
- Last close price and datetime for reference

```bash
python mt5/scripts/forecast.py
# For a single instrument:
python mt5/scripts/forecast.py --instrument XAUUSD --horizon 48
# Without the fine-tuned adapter (base model only):
python mt5/scripts/forecast.py --no_adapter
```

### 4.6 Step 7 — Export to MetaTrader 5

The `export_to_mt5.py` script writes per-instrument forecast CSVs into the MetaTrader 5 terminal's `MQL5/Files/TimesFM/` directory:

```python
def find_mt5_files_dir() -> Path:
    """Auto-detect MT5 terminal's MQL5/Files directory."""
    if mt5 is not None:
        if mt5.initialize():
            info = mt5.terminal_info()
            files_dir = Path(info.data_path) / "MQL5" / "Files"
            mt5.shutdown()
            return files_dir
    # Fallback: scan AppData/Roaming/MetaQuotes/Terminal/
    ...
```

The CSV format is designed for easy parsing by the MQL5 indicator:

```csv
datetime,point,q10,q50,q90
2026.04.08 14:00,1.0842,1.0831,1.0841,1.0853
2026.04.08 15:00,1.0845,1.0830,1.0844,1.0860
...
```

A `last_update.txt` timestamp file is also written so the indicator can display when the forecast was last refreshed.

```bash
python mt5/scripts/export_to_mt5.py
```


## 5. The MQL5 Indicator

The `TimesFM_Forecast.mq5` indicator reads the CSV files produced by Step 7 and renders the forecast directly on the MetaTrader 5 chart. Unlike standard indicators that use indicator buffers, this one draws chart objects — which is necessary because the forecast extends into the future, beyond the last formed bar.

### 5.1 Structure

The indicator creates four visual elements:

1. **Point forecast line** — solid trend lines colored green (bullish) or red (bearish) based on the direction from the last close to the final forecast point
2. **90% confidence band** — filled triangles between the q10 and q90 quantile lines, providing visual uncertainty estimation
3. **Median line (q50)** — dotted white line showing the median forecast
4. **Reference line** — dashed horizontal line at the last close price for easy comparison

### 5.2 CSV Loading

The `LoadForecastCSV()` function parses the forecast file row by row:

```mql5
bool LoadForecastCSV()
{
   string filename = InpSubfolder + "\\" + g_symbol + "_forecast.csv";
   int handle = FileOpen(filename, FILE_READ | FILE_CSV | FILE_ANSI, ',');
   if(handle == INVALID_HANDLE)
      return false;

   // Skip header
   while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
      FileReadString(handle);

   // Read data rows into arrays
   while(!FileIsEnding(handle))
   {
      string dt_str = FileReadString(handle);
      string pt_str = FileReadString(handle);
      string q10_str = FileReadString(handle);
      string q50_str = FileReadString(handle);
      string q90_str = FileReadString(handle);
      // ... convert and store in global arrays
   }
   FileClose(handle);
   return true;
}
```

### 5.3 Drawing the Forecast

The confidence band is rendered as pairs of filled triangles for each segment between adjacent forecast points:

```mql5
for(int i = 0; i < g_horizon - 1; i++)
{
   datetime t1 = g_forecast_times[i];
   datetime t2 = g_forecast_times[i+1];

   // Triangle 1: upper-left, upper-right, lower-left
   CreateBandTriangle(OBJ_PREFIX + "band1_" + IntegerToString(i),
      t1, g_q90[i],  t2, g_q90[i+1],  t1, g_q10[i],  InpBandColor);

   // Triangle 2: upper-right, lower-right, lower-left
   CreateBandTriangle(OBJ_PREFIX + "band2_" + IntegerToString(i),
      t2, g_q90[i+1],  t2, g_q10[i+1],  t1, g_q10[i],  InpBandColor);
}
```

Each quad (between two time points) is decomposed into two triangles because MQL5's `OBJ_TRIANGLE` type only supports three anchor points.

A direction label is displayed in the upper-left corner showing the forecast signal, percentage change, horizon, and last update time:

```
TimesFM: LONG +0.42% | 48h | Updated: 2026.04.08 13:00:00
```

### 5.4 Auto-Reload

The indicator automatically reloads the CSV on each new bar (if `InpAutoReload` is enabled) and also supports manual reload by pressing the **R** key:

```mql5
void OnChartEvent(const int id, const long &lparam,
                  const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_KEYDOWN && lparam == 'R')
   {
      if(LoadForecastCSV())
      {
         g_loaded = true;
         DrawForecast();
      }
   }
}
```

### 5.5 Installation

Copy the indicator file into your MetaTrader 5 data directory:

1. Open MetaTrader 5
2. Go to **File → Open Data Folder**
3. Navigate to `MQL5/Indicators/`
4. Copy `TimesFM_Forecast.mq5` into this folder
5. In the Navigator panel, right-click **Indicators** → **Refresh**
6. Compile the indicator (double-click to open in MetaEditor, then press **F7**)
7. Drag the indicator onto any chart whose symbol has a forecast CSV

### 5.6 Input Parameters

| Parameter | Default | Description |
|---|---|---|
| `InpBullColor` | Lime | Color for bullish forecast lines |
| `InpBearColor` | Tomato | Color for bearish forecast lines |
| `InpBandColor` | DodgerBlue | Fill color for confidence band |
| `InpMedianColor` | White | Color for median (q50) line |
| `InpForecastWidth` | 3 | Width of the forecast line |
| `InpBandWidth` | 1 | Width of band border lines |
| `InpAutoReload` | true | Reload forecast on each new bar |
| `InpSubfolder` | "TimesFM" | Subfolder name in MQL5/Files/ |


## 6. Visualization Dashboard

For those who prefer analyzing all instruments at once outside of MetaTrader 5, the `visualize_forecast.py` script generates an interactive HTML dashboard using Plotly:

```bash
python mt5/scripts/visualize_forecast.py
# Custom history window:
python mt5/scripts/visualize_forecast.py --history 120 --output my_dashboard.html
```

The dashboard shows:

- Recent price history (gray) for each instrument
- Point forecast with directional coloring (green/red)
- 90% confidence band as a filled region
- Horizontal reference line at the last close
- A summary table ranking instruments by "conviction" — the ratio of expected move to uncertainty width

The output is saved to `mt5/data/forecast_dashboard.html` and can be opened in any browser.


## 7. Running the Full Pipeline

With everything installed and configured, the complete pipeline can be executed with a single command:

```bash
python mt5/run_pipeline.py
```

This sequentially runs all 7 steps:

```
Step 1/7: Export OHLCV from MetaTrader 5
Step 2/7: Generate moon phase features
Step 3/7: Generate economic calendar
Step 4/7: Build merged covariates
Step 5/7: Fine-tune TimesFM (LoRA)
Step 6/7: Forecast all instruments
Step 7/7: Export forecasts to MT5
```

Common workflow shortcuts:

```bash
# Skip fine-tuning (reuse existing adapter)
python mt5/run_pipeline.py --skip-finetune

# Skip moon phases (they don't change)
python mt5/run_pipeline.py --skip-moon

# Run only forecast and export (after data is ready)
python mt5/run_pipeline.py --only 6 7

# Resume from covariate building (e.g., after adding a new instrument)
python mt5/run_pipeline.py --from-step 4
```


## 8. Practical Considerations

### 8.1 Overfitting Defenses

Financial data is uniquely challenging for machine learning — the signal-to-noise ratio is extremely low and the underlying process is non-stationary. Our pipeline employs multiple layers of defense:

1. **Minimal adapter capacity** — LoRA rank 4 with only 2 adapted layers (102K params vs. 200M frozen)
2. **Chronological train/val split** — No data leakage from future to past
3. **Strong regularization** — Weight decay 0.1 + LoRA dropout 0.15
4. **Early stopping** — Halts training when validation loss stops improving (patience=15)
5. **Conservative learning rate** — 5e-5 with 15% linear warmup

### 8.2 When to Re-Train

The adapter should be retrained when:
- Significant new data accumulates (every few weeks for H1 timeframe)
- Market regime changes visibly (e.g., transition from low to high volatility)
- Forecast accuracy degrades notably

The pipeline makes this trivial: run `python mt5/run_pipeline.py --from-step 1` to pull fresh data and retrain.

### 8.3 Limitations

- **Not a standalone trading signal.** The forecasts should be combined with risk management, position sizing, and confirmation from other analysis methods
- **Forecast quality degrades with horizon.** The default 48-bar (2-day) horizon is already aggressive for financial data. Longer horizons should be treated with lower confidence
- **Covariate forward projection is approximate.** Future values of technical features (ATR, volume) are forward-filled from the last known value, which is a simplification
- **Latency.** The Python pipeline is not built for high-frequency trading. It is designed for intraday (H1) or higher timeframes


## Conclusion

In this article, we have built a complete bridge between Google's TimesFM 2.5 foundation model and MetaTrader 5. The key elements of the system include:

- **Data export** from MetaTrader 5 covering 14 instruments across forex, indices, and commodities
- **Rich covariate engineering** incorporating astronomical data, macroeconomic events, market sessions, and technical features
- **Parameter-efficient fine-tuning** using LoRA adapters, carefully configured to prevent overfitting on non-stationary financial data
- **Probabilistic forecasting** with quantile estimates providing uncertainty bands
- **Bidirectional MT5 integration** — data flows out of MetaTrader 5 for training and back in for chart visualization

The pretrained TimesFM model brings an understanding of temporal patterns learned from 100 billion data points, while the LoRA adaptation specializes this knowledge for financial markets at minimal overfitting risk. The MQL5 indicator provides visual feedback directly in the trading terminal, making the forecasts actionable.

This work demonstrates that foundation models — originally developed for natural language — can be effectively repurposed for financial time series when paired with appropriate fine-tuning strategies and domain-specific feature engineering. As these models continue to grow in capability, the gap between zero-shot and supervised performance will likely narrow further, making transfer learning an increasingly practical tool in the algorithmic trader's arsenal.


## List of References

1. A. Das, W. Kong, A. Leach, S. K. Mathur, R. Sen, R. Yu. ["A decoder-only foundation model for time-series forecasting"](https://arxiv.org/abs/2310.10688), ICML 2024
2. [TimesFM GitHub Repository](https://github.com/google-research/timesfm) — Google Research
3. [TimesFM 2.5 200M PyTorch Model](https://huggingface.co/google/timesfm-2.5-200m-pytorch) — HuggingFace
4. [Google Research Blog: A decoder-only foundation model for time-series forecasting](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
5. [TimesFM in BigQuery ML](https://docs.cloud.google.com/bigquery/docs/timesfm-model) — Google Cloud Documentation
6. E. J. Hu et al. ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685), ICLR 2022
7. Y. Nie et al. ["A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"](https://arxiv.org/abs/2211.14730) (PatchTST)


## Programs Used in the Article

| # | File | Type | Description |
|---|---|---|---|
| 1 | config.py | Configuration | All paths, instruments, hyperparameters, and defaults |
| 2 | run_pipeline.py | Orchestrator | Runs the full 7-step pipeline with CLI controls |
| 3 | export_mt5_data.py | Script | Exports OHLCV data from MetaTrader 5 |
| 4 | generate_moon_phases.py | Script | Computes hourly moon phases using PyEphem |
| 5 | generate_economic_calendar.py | Script | Fetches economic event dates from FRED API |
| 6 | build_covariates.py | Script | Merges all features into per-instrument CSVs |
| 7 | finetune.py | Script | Fine-tunes TimesFM 2.5 with LoRA/DoRA adapters |
| 8 | forecast.py | Script | Generates probabilistic forecasts for all instruments |
| 9 | export_to_mt5.py | Script | Writes forecast CSVs into MQL5/Files/ directory |
| 10 | visualize_forecast.py | Script | Generates interactive HTML forecast dashboard |
| 11 | TimesFM_Forecast.mq5 | Indicator | MQL5 chart indicator displaying forecast with confidence bands |

All Python scripts are located in `mt5/scripts/`. The MQL5 indicator is in `mt5/mql5/`. The full `mt5/` folder is available as a [ZIP download](mt5.zip) attached to this article.
