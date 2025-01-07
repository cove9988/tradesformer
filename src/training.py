# %%
from src.ppo_model import single_csv_training, multiply_csv_files_traning

features = ['open', 'high', 'low', 'close', 'vol', 'macd','boll_ub','boll_lb','rsi_30','dx_30','close_30_sma','close_60_sma']

csv_file = "/home/paulg/github/tradesformer/data/split/AUDUSD/weekly/AUDUSD_2022_1.csv"
single_csv_training(csv_file=csv_file)


# data_directory = "/home/paulg/github/tradesformer/data/split/AUDUSD/weekly"


# %%
