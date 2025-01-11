# %%
import glob
import os
import pandas as pd
import time
import datetime
from stable_baselines3 import PPO
from src.ppo_model import ForexTradingEnv, load_data
import logging
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

features = ['open', 'high', 'low', 'close', 'minute', 'hour', 'day', 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma']
model_file = '/home/paulg/github/tradesformer/data/model/AUDUSD/weekly/AUDUSD_2023_66.zip'
# csv_file = "/home/paulg/github/tradesformer/data/split/EURUSD/weekly/EURUSD_2024_103.csv"
data_directory = "/home/paulg/github/tradesformer/data/split/AUDUSD/weekly"
csv_files = glob.glob(os.path.join(data_directory, "*.csv"))
run_time = 10
_run = 1
for file in csv_files :
    if _run > run_time: break
    # Read the CSV file
    env = ForexTradingEnv(file,features)
    model = PPO.load(model_file, env=env)
# %%
    observation, info = env.reset()
    done = False
    total_buy = 0
    total_sell = 0
    totoal_rewards = 0
    while not done:
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.render(mode = 'graph')
    print(f'------rewards:{totoal_rewards}-----buy:{total_buy}--sell:{total_sell}------')
    _run += 1



# %%
