# %%
print('start...1.')
from alpha_vantage.foreignexchange import ForeignExchange
import pandas as pd

print('start....')
api_key = 'ZT9348FDNFPIUEFZ'
cc = ForeignExchange(key=api_key)
pair="EURUSD"
interval='5min'
# Fetch intraday data'
data, _ = cc.get_currency_exchange_intraday(from_symbol='EUR', to_symbol='USD', interval=interval, outputsize='full')

# Convert to DataFrame
df = pd.DataFrame.from_dict(data, orient='index')
df.index = pd.to_datetime(df.index)
df = df.sort_index()
filename = f'data/alpha-{pair}-{interval}.csv'
data.to_csv(filename)
print(f'Data saved to {filename}')
# %%
from alpha_vantage.foreignexchange import ForeignExchange
from pprint import pprint
cc = ForeignExchange(key='ZT9348FDNFPIUEFZ')
# There is no metadata in this call
data, _ = cc.get_currency_exchange_rate(from_currency='BTC',to_currency='USD')
pprint(data)
# %%
