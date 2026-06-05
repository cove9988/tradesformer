# data_processor.py
import os
import sys
import logging
import pandas as pd
import numpy as np
from src.timesfm_features import (
    TIMESFM_HELPER_COLUMNS,
    TIMESFM_OBSERVATION_COLUMNS,
    add_causal_timesfm_features,
    add_time_context_features,
    validate_timesfm_week,
)
from src.util.read_config import EnvConfig
from src.util.logger_config import setup_logging

logger = logging.getLogger(__name__)


def patch_missing_data(df, dt_col_name='time', cf=None):
    if cf is None:
        raise ValueError("EnvConfig is required for patch_missing_data")

    required_cols = cf.data_processing_parameters("required_cols")
    if df.shape[1] == 6:
        df.columns = required_cols + ['vol']
    elif df.shape[1] == 5:
        df.columns = required_cols
    else:
        raise ValueError(f"Invalid number of columns: {df.shape[1]} => {required_cols}")

    if missing := set(required_cols) - set(df.columns):
        raise ValueError(f"Missing columns: {missing}")

    dt_candidates = {'time', 'timestamp', 'date', 'datetime'}
    if dt_col_name not in df.columns:
        found = list(dt_candidates & set(df.columns))
        if not found:
            raise KeyError(f"No datetime column found. Tried: {dt_candidates}")
        dt_col_name = found[0]
        logger.info(f"Using datetime column: {dt_col_name}")

    df[dt_col_name] = pd.to_datetime(df[dt_col_name], utc=True)
    df = df.set_index(dt_col_name).sort_index()

    new_index = pd.date_range(
        start=df.index.min().floor('D'),
        end=df.index.max().ceil('D'),
        freq='5min',
        tz='UTC'
    )

    df = df.reindex(new_index)
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill()

    df = df[(df.index.weekday < 5) | (
        (df.index.weekday == 4) & (df.index.hour >= 22)
    )]

    min_bars = cf.data_processing_parameters("min_bars_per_week")
    for week, group in df.groupby(pd.Grouper(freq='W-MON')):
        if len(group) != min_bars:
            logger.warning(f"Week {week} has {len(group)}/{min_bars} bars")

    return df


def build_timesfm_dataset(df, symbol, cf):
    point = cf.symbol(symbol, "point")
    time_context = add_time_context_features(df, symbol=symbol)
    feature_df = add_causal_timesfm_features(time_context, point=point)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    return feature_df


def split_timesfm_weeks(df, symbol='EURUSD', cf=None):
    if cf is None:
        raise ValueError("EnvConfig is required for split_timesfm_weeks")

    split_cfg = cf.data_processing_parameters("train_eval_split")
    base_path = split_cfg["base_path"].format(symbol=symbol)
    min_bars = cf.data_processing_parameters("min_bars_per_week")

    groups = df.groupby(pd.Grouper(freq='W-MON'))

    for week_start, week_df in groups:
        if week_df.empty:
            continue

        is_valid, reason = validate_timesfm_week(
            week_df,
            bars_per_week=min_bars,
            feature_cols=TIMESFM_OBSERVATION_COLUMNS,
        )
        if not is_valid:
            logger.warning(f"Skipping {week_start}: {reason}")
            continue

        path = os.path.join(base_path, split_cfg["train_dir"])
        os.makedirs(path, exist_ok=True)

        iso_year, iso_week, _ = week_start.isocalendar()
        fname = f"{symbol}_{iso_year}_{iso_week:02d}.csv"
        keep_cols = [
            'open', 'high', 'low', 'close', 'symbol',
            *TIMESFM_OBSERVATION_COLUMNS,
            *TIMESFM_HELPER_COLUMNS,
        ]
        export_df = week_df[keep_cols].reset_index().rename(columns={'index': 'time'})
        export_df.to_csv(f"{path}/{fname}", index=False)
        logger.info(f"Saved TimeFM weekly file: {fname}")


    # Backward-compatible TimeFM aliases.
    build_timefm_dataset = build_timesfm_dataset
    split_timefm_weeks = split_timesfm_weeks


if __name__ == '__main__':
    try:
        if len(sys.argv) == 2:
            symbol = sys.argv[1]
            file = f'./data/raw/{symbol}_M5.csv'
        elif len(sys.argv) == 4:
            symbol, _, file = sys.argv[1], sys.argv[2], sys.argv[3]
        else:
            raise ValueError("Usage: python -m src.data_processor SYMBOL [FREQ INPUT_CSV]")

        cf = EnvConfig('./src/configure.json')
        setup_logging(asset=symbol, console_level=logging.INFO, file_level=logging.INFO)
        logger.info(f"Processing TimeFM dataset for {symbol} from {file}")

        df = pd.read_csv(file)
        df = patch_missing_data(df, cf=cf)
        df = build_timesfm_dataset(df, symbol=symbol, cf=cf)
        split_timesfm_weeks(df, symbol=symbol, cf=cf)
        logger.info("TimeFM processing completed successfully")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        sys.exit(1)
