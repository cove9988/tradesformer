# %%
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define the currency pair and period
pair = 'EURUSD'
symbol = 'EURUSD=X'  # EUR/USD exchange rate
end_date = datetime.today()
start_date = end_date - timedelta(days=59)  # Last 60 days
# %%
start_date
# %%
interval = '5m'  # 5-minute intervals
end_date = datetime.today()
start_date = end_date - timedelta(days=60)
data =''
for i in range(0, 10):
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    data = data.dropna()
    filename = f'data/yahoo-{pair}-{start_date.strftime("%Y-%m-%d")}-{end_date.strftime("%Y-%m-%d")}-{interval}.csv'
    data.to_csv(filename)
    print(f'Data saved to {filename}')
    end_date = end_date - timedelta(days=60)
    start_date = end_date - timedelta(days=60)
# %%
