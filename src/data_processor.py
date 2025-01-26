# data_processor.py
import os
import sys
import logging
import pandas as pd
import numpy as np
from finta import TA
from sklearn.preprocessing import StandardScaler
from src.util.read_config import EnvConfig
from src.util.logger_config import setup_logging

# Configure logging
logger = logging.getLogger(__name__)

def patch_missing_data(df, dt_col_name='time', cf=None):
    required_cols = cf.data_processing_parameters("required_cols")    
    print(required_cols) 
    if df.shape[1] == 6:
        df.columns = required_cols + ['vol']  
    elif df.shape[1] != 5:
        df.columns = required_cols
    else:
        raise ValueError(f"Invalid number of columns: {df.shape[1]} =>{required_cols}")
    
    # 1. Column validation
    if missing := set(required_cols) - set(df.columns):
        raise ValueError(f"Missing columns: {missing}")

    # 2. Auto-detect datetime column
    dt_candidates = {'time', 'timestamp', 'date', 'datetime'}
    if dt_col_name not in df.columns:
        found = list(dt_candidates & set(df.columns))
        if not found:
            raise KeyError(f"No datetime column found. Tried: {dt_candidates}")
        dt_col_name = found[0]
        logger.info(f"Using datetime column: {dt_col_name}")

    # 3. Convert to datetime index
    df[dt_col_name] = pd.to_datetime(df[dt_col_name], utc=True)
    df = df.set_index(dt_col_name).sort_index()

    # 4. Create complete 5-min grid (Mon 00:00 - Fri 23:55 UTC)
    new_index = pd.date_range(
        start=df.index.min().floor('D'),
        end=df.index.max().ceil('D'),
        freq='5T',
        tz='UTC'
    )
    
    # 5. Forward-fill OHLC prices
    df = df.reindex(new_index)
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill()
    
    # 6. Filter weekends (keep Friday 22:00-23:55 as "pseudo Sunday")
    df = df[(df.index.weekday < 5) | (
        (df.index.weekday == 4) & (df.index.hour >= 22)
    )]

    # 7. Validate bars per week
    min_bars = cf.data_processing_parameters("min_bars_per_week")
    for week, group in df.groupby(pd.Grouper(freq='W-MON')):
        if len(group) != min_bars:
            logger.warning(f"Week {week} has {len(group)}/{min_bars} bars")
    
    return df.reset_index().rename(columns={'index': dt_col_name})

def add_time_feature(df, symbol):
    """Add temporal features with proper index handling"""
    
    if 'time' not in df.columns:
        raise KeyError("'time' column missing after patch_missing_data")
        
    df = df.set_index('time')
    df.index = pd.to_datetime(df.index, utc=True)
    
    # Cyclical time features
    df['weekday'] = df.index.dayofweek  # 0=Monday
    df['hour'] = df.index.hour
    df['minute_block'] = df.index.minute // 5  # 0-11
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24).round(6)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24).round(6)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute_block']/12).round(6)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute_block']/12).round(6)

    # Market sessions (GMT)
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
    df['overlap_session'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
    
    df['symbol'] = symbol
    return df.reset_index()

def tech_indicators(df, cf=None):  # 288 = 24hrs in 5-min bars
    """Calculate technical indicators with proper NaN handling"""
    period = cf.data_processing_parameters("indicator_period")
    scale_cols = cf.data_processing_parameters("scale_cols")    
    # Calculate indicators
    df['macd'] = TA.MACD(df).SIGNAL.ffill().round(6)
    bb = TA.BBANDS(df)
    df['boll_ub'] = bb['BB_UPPER'].ffill()
    df['boll_lb'] = bb['BB_LOWER'].ffill()
    
    df['rsi_30'] = TA.RSI(df, period=period).ffill()
    df['dx_30'] = TA.ADX(df, period=period).ffill()
    df['close_30_sma'] = TA.SMA(df, period=period).ffill()
    df['close_60_sma'] = TA.SMA(df, period=period*2).ffill()
    df['atr'] = TA.ATR(df, period=period).ffill()
     # Add returns and volatility ratio
    df['returns_5'] = df['close'].pct_change(5).round(6)
    df['returns_24'] = df['close'].pct_change(24).round(6)
    df['volatility_ratio'] = (df['high'] - df['low']) / df['close'].round(6)
        
    # Normalize
    scaler = StandardScaler()
    scale_cols = ['macd', 'rsi_30', 'atr', 'dx_30']
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    # 1. Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # 2. Apply clipping only to numeric features
    df[numeric_cols] = df[numeric_cols].clip(lower=-1e5, upper=1e5)
    # 3. Round decimal values
    df[numeric_cols] = df[numeric_cols].round(6)  
    return df

def split_timeserious(df, freq='W', symbol='EURUSD',cf=None):
    """Split data with proper weekly alignment"""
    split_cfg = cf.data_processing_parameters("train_eval_split")
    base_path = split_cfg["base_path"].format(symbol=symbol)
        
    # Align with Forex week (Monday-Sunday)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time')
    
    groups = df.groupby(pd.Grouper(freq='W-MON'))
    
    for week_start, week_df in groups:
        if week_df.empty:
            continue
        
        # 1. Check raw indicators before normalization
        indicator_cols = ['macd', 'boll_ub', 'boll_lb',
                         'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'atr']
        first_row = week_df[indicator_cols].iloc[0]
        has_nan = first_row.isna().any()
        has_zero = (first_row == 0).any()
        is_eval = has_nan or has_zero

        # 2. Normalize and validate
        week_df = normalize_features(week_df)
        if len(week_df) < 1440:
            logger.warning(f"Skipping {week_start}: {len(week_df)}/1440 bars")
            continue

        # 3. Save to appropriate directory
        dir_type = 'eval' if is_eval else 'train'
        path =  os.path.join(base_path, split_cfg[f"{dir_type}_dir"])
        os.makedirs(path, exist_ok=True)
        
        iso_year, iso_week, _ = week_start.isocalendar()
        fname = f"{symbol}_{iso_year}_{iso_week:02d}.csv"
        week_df.reset_index().to_csv(f"{path}/{fname}", index=False)
        logger.critical(f"Saved {dir_type} file: {fname}")

def normalize_features(df):
    """Z-score normalization instead of pivot-based"""
    price_cols = ['open', 'high', 'low', 'close']
    df[price_cols] = ((df[price_cols] - df[price_cols].mean()) / df[price_cols].std()).round(6)
    return df

if __name__ == '__main__':
    try:
        if len(sys.argv) == 2:
            symbol = sys.argv[1]
            freq = 'W'
            file = f'./data/raw/{symbol}_M5.csv'
        else:
            symbol, freq, file = sys.argv[1], sys.argv[2], sys.argv[3]
            
        cf = EnvConfig('./src/configure.json')  
        setup_logging(asset=symbol, console_level=logging.INFO, file_level=logging.INFO)
        logger.info(f"Processing {symbol} {freq} data from {file}")
        # 1. Load & clean
        df = pd.read_csv(file)
        df = patch_missing_data(df,cf=cf)
        
        # 2. Feature engineering
        df = add_time_feature(df, symbol=symbol)
        df = tech_indicators(df, cf=cf) 
        
        # 3. Split & save
        split_timeserious(df, freq=freq, symbol=symbol, cf=cf)
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        sys.exit(1)