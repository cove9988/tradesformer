import os
import sys
import pandas as pd
from finta import TA

def patch_missing_data(df, dt_col_name):
    if df.shape[1] == 6:
        df.columns = ['time', 'open', 'high', 'low', 'close', 'vol']    
    
    # df["time"] = pd.to_datetime(df["time"])
    # df.set_index("time", inplace=True)
    df['dt'] = pd.to_datetime(df[dt_col_name])
    df.index = df['dt']    
    df["weekday"] = df.index.dayofweek  # Monday=0, Sunday=6
    df["original_date"] = df.index          # Keep the original date for reference
    df.index = df.index.where(df["weekday"] != 6, df.index - pd.Timedelta(days=2))
    df["time"] = df.index
    
    # sunday_rows = df[df["weekday"] == 6].copy()
    # sunday_rows["adjusted_timestamp"] = sunday_rows.index - pd.Timedelta(days=2)
    # print(sunday_rows[:100])

    # Drop helper columns if not needed
    df.drop(columns=["original_date","weekday"], inplace=True)
    df = df.sort_index()
   # Check for duplicate index values
    if not df.index.is_unique:
        print("Duplicate index values detected. Resolving...")
        # Drop duplicate indices and keep the first occurrence
        df = df[~df.index.duplicated(keep='first')]
            
    # Generate the complete time range
    full_time_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="5min")
    full_time_range = full_time_range[full_time_range.weekday < 5]
    # Identify missing timestamps
    missing_timestamps = full_time_range.difference(df.index)

    # Insert missing rows with the previous row's data
    for ts in missing_timestamps:
        print(f"Missing timestamp: {ts}")
        # Find the closest previous row using .asof()
        previous_index = df.index.asof(ts)
        if previous_index is None:
            print(f"Error: No earlier data available for missing timestamp {ts}")
            continue

        previous_row = df.loc[previous_index]
        new_row = previous_row.copy()
        new_row["time"] = ts  # Assign the missing timestamp
        new_row["dt"] = ts
        new_row.name = ts
        # print(f"-------------\n{new_row}\n--------")
        df = pd.concat([df, pd.DataFrame([new_row])])

        # Sort the DataFrame by datetime
        df.sort_index(inplace=True)


    # df.rename(columns={"index": "time"}, inplace=True)
    return df


def add_time_feature(df,symbol):
    """read csv into df and index on time
    dt_col_name can be any unit from minutes to day. time is the index of pd
    must have pd columns [(time_col),(asset_col), open,close,high,low,day]
    data_process will add additional time information: time(index), minute, hour, weekday, week, month,year, day(since 1970)
    use StopLoss and ProfitTaken to simplify the action,
    feed a fixed StopLoss (SL = 200) and PT = SL * ratio
    action space: [action[0,2],ratio[0,10]]
    rewards is point
    
    add hourly, dayofweek(0-6, Sun-Sat)
    Args:
        file (str): file path/name.csv
    """

    df['symbol'] = symbol
    # df.index.names = ['step']
    df["weekday"] = df.index.dayofweek
    df['minute'] =df['dt'].dt.minute
    df['hour'] =df['dt'].dt.hour
    df['week'] = df['dt'].dt.isocalendar().week
    df['month'] = df['dt'].dt.month
    df['year'] = df['dt'].dt.year
    df['day'] = df['dt'].dt.day
    # df = df.set_index('dt')
    return df 

# 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30','close_30_sma', 'close_60_sma'
def tech_indictors(df):
    df['macd'] = TA.MACD(df).SIGNAL
    df['boll_ub'] = TA.BBANDS(df).BB_UPPER
    df['boll_lb'] = TA.BBANDS(df).BB_LOWER
    df['rsi_30'] = TA.RSI(df,period= 30)
    df['dx_30'] = TA.ADX(df,period= 30)
    df['close_30_sma'] = TA.SMA(df,period=30)
    df['close_60_sma'] = TA.SMA(df,period=60)        
    
    #fill NaN to 0
    df = df.fillna(0)
    print(f'--------df head - tail ----------------\n{df.head(3)}\n{df.tail(3)}\n---------------------------------')
    
    return df 
    
def split_timeserious(df, key_ts='dt', freq='W', symbol=''):
    """import df and split into hour, daily, weekly, monthly based and 
    save into subfolder

    Args:
        df (pandas df with timestamp is part of multi index): 
        spliter (str): H, D, W, M, Y
    """

    df = df.set_index(key_ts)
    freq_name = {'H':'hourly','D':'daily','W':'weekly','M':'monthly','Y':'Yearly'}
    count = 0
    for n, g in df.groupby(pd.Grouper(level=key_ts,freq=freq)):
        p =f'./data/split/{symbol}/{freq_name[freq]}'
        os.makedirs(p, exist_ok=True)
        #fname = f'{symbol}_{n:%Y%m%d}_{freq}_{count}.csv'
        fname = f'{symbol}_{n:%Y}_{count}.csv'
        fn = f'{p}/{fname}'
        print(f'save to:{fn} -- row {len(g)}')
        g.reset_index(drop=True, inplace=True)
        # g.drop(columns =['dt'], inplace=True)
        g.to_csv(fn)
        count += 1
    return 
"""
python ./data/data_processor.py GBPUSD W ./data/raw/GBPUSD_M5.csv
symbol="GBPUSD"
freq = [H, D, W, M]
file .csv, column names [time, open, high, low, close, vol]
"""
if __name__ == '__main__':
    symbol, freq, file = sys.argv[1],sys.argv[2],sys.argv[3]
    print(f'processing... symbol:{symbol} freq:{freq} file:{file}')
    try :
        df = pd.read_csv(file)
    except Exception:
        print(f'No such file or directory: {file}') 
        exit(0)
    df = patch_missing_data(df,dt_col_name='time')            
    df = add_time_feature(df, symbol=symbol)
    df = tech_indictors(df)
    split_timeserious(df,freq=freq, symbol=symbol)