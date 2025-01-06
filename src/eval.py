# %%
import pandas as pd
import time
import datetime
from stable_baselines3 import PPO
from ppo_model import ForexTradingEnv, load_data
features = ['open', 'high', 'low', 'close', 'vol', 'macd','boll_ub','boll_lb','rsi_30','dx_30','close_30_sma','close_60_sma']
model_file = '/home/paulg/github/tradesformer/data/model/ppo_forex_transformer_batch_138.zip'
csv_file = "/home/paulg/github/tradesformer/data/split/AUDUSD/weekly/AUDUSD_2024_122.csv"

data = load_data(csv_file, features)

# %%

env = ForexTradingEnv(data,features)
# %%
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
    if action or reward :
        print(f'{env.current_step} {action},reward:{reward}')    
    if reward != 0:
        totoal_rewards += reward
    if action == 1:
        total_buy += 1
    if action == 2:
        total_sell += 1        
    done = terminated or truncated
    # env.render()
print(f'------rewards:{totoal_rewards}-----buy:{total_buy}--sell:{total_sell}------')




# %%
