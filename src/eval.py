# %%
import glob
import os
import pandas as pd
import time
import datetime
import logging
from stable_baselines3 import PPO
from src.ppo_model import ForexTradingEnv
from src.util.read_config import EnvConfig
from src.util.logger_config import setup_logging

logger = logging.getLogger(__name__)

def eval(data_directory, env_config_file, model_file, asset, run_time = 10, mode = 'graph', sequence_length=24):
    csv_files = glob.glob(os.path.join(data_directory, "*.csv"))
    cf = EnvConfig(env_config_file)
    features = cf.env_parameters("observation_list")
    print(features)

    _run = 1
    for file in csv_files :
        if _run > run_time: break
        # Read the CSV file
        env = ForexTradingEnv(file,cf,asset,features,sequence_length)
        model = PPO.load(model_file, env=env)
    # %%
        observation, info = env.reset()
        done = False
        total_buy = 0
        total_sell = 0
        total_rewards = 0
        step = 0
        while not done:
            action, _states = model.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f'step:{step} rwd:{reward} action:{action} ')
            step += 1
            total_rewards += reward
            if action == 1: total_buy += 1
            if action == 2: total_buy += 1
        env.render(mode = mode)
        print(f'------rewards:{total_rewards}-----buy:{total_buy}--sell:{total_sell}------')
        _run += 1

if __name__ == "__main__":
    asset = "AUDUSD"   
    env_config_file = '/home/paulg/github/tradesformer/src/configure.json'
    model_file = '/home/paulg/github/tradesformer/data/model/AUDUSD/weekly/AUDUSD_2024_120.zip'
    data_directory = "/home/paulg/github/tradesformer/data/split/AUDUSD/weekly"
    setup_logging(asset=asset, console_level=logging.ERROR, file_level=logging.INFO)
    eval(data_directory, env_config_file, model_file, asset, run_time= 5, mode = 'graph', sequence_length=24)
