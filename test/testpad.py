# %%
import pandas as pd

csv_file = "/home/paulg/github/tradesformer/data/split/AUDUSD/weekly/AUDUSD_2022_1.csv"
data = pd.read_csv(csv_file)
#%%
data
# %%
current_step= 30
_o, _c, _h, _l,_t,_day = data.iloc[current_step][["open","close","high","low","time","day"]]
# %%
position={}
entry_price = _c
position['ActionPrice'] = _c
direction = "Buy"
position['PT'] = 500
position['SL'] = -200
point=100000

# %%
current_step= 500
_o, _c, _h, _l,_t,_day = data.iloc[current_step][["open","close","high","low","time","day"]]
profit_target_price = entry_price + position['PT']/point if direction == 'Buy' else entry_price - position['PT']/point
stop_loss_price = entry_price + position['SL']/point if direction == 'Buy' else entry_price - position['SL']/point
print(_c, stop_loss_price, profit_target_price)
print(_l, stop_loss_price, _l <= stop_loss_price )
# %%
_o, _c, _h, _l,_t,_day

# %%
