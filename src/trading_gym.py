import math
import datetime
import random
import json
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from util.plot_chart import TradingChart
from util.log_render import render_to_file
from util.read_config import EnvConfig
from src.util.action_agg import ActionEnum, form_action
from util.rewards import RewardCalculator
from util.transaction import TransactionManager

class tgym(gym.Env):
    """
    Rewards 
        value[step...forward_window]   BUY     SELL    HOLD
        SL                             SL      SL      +
        PT                             PT      PT      -
        NN                             -       -       +
    """
    metadata = {'render.modes': ['graph', 'human', 'file']}
    env_id = "TradingGym-v9"

    def __init__(self, df, env_config_file='./env/config/gdbusd-test-9.json') -> None:
        super(tgym, self).__init__()
        self.cf = EnvConfig(env_config_file)
        self.balance_initial = self.cf.env_parameters("balance")
        self.asset_col = self.cf.env_parameters("asset_col")
        self.shaping_reward = self.cf.env_parameters("shaping_reward")
        self.stop_loss = self.cf.symbol(self.cf.env_parameters("asset_col"), "stop_loss_max")
        self.profit_taken = self.cf.symbol(self.cf.env_parameters("asset_col"), "profit_taken_max")
        self.backward_window = self.cf.env_parameters("backward_window")
        self.df = df

        self.reward_calculator = RewardCalculator(
            df, self.cf, self.shaping_reward, self.stop_loss, self.profit_taken, self.backward_window
        )
        self.transaction_manager = TransactionManager(
            self.cf, self.balance_initial, self.cf.env_parameters("asset_col"), self.stop_loss, self.profit_taken
        )

        self.reset()

    def step(self, action):
        self.current_step += 1
        done = (self.transaction_manager.balance <= 0 or self.current_step == len(self.df) - 1)
        reward = self.reward_calculator.reward_box[self.current_step - self.backward_window][action]

        if not self.transaction_manager._close_order(self.df, self.current_step, done):
            reward = 0

        return self.get_observation(), reward, done, {
            "Close": self.transaction_manager.transaction_close_this_step
        }

    def reset(self):
        self.current_step = self.backward_window
        self.transaction_manager.balance = self.balance_initial
        self.transaction_manager.transaction_live = []
        self.transaction_manager.transaction_history = []
        return self.get_observation()

    def get_observation(self):
        if self.current_step - self.backward_window < 0:
            return []
        else:
            _d = self.df.iloc[self.current_step - self.backward_window:self.current_step]
            return _d[self.cf.env_parameters("observation_list")].to_numpy()

    def render(self, mode='human', title=None, **kwargs):
        if mode in ('human', 'file'):
            render_to_file(log_filename=self.cf.env_parameters("log_filename"), balance=self.transaction_manager.balance)
        elif mode == 'graph':
            p = TradingChart(self.df, self.transaction_manager.transaction_history)
            p.plot()

    def close(self):
        pass

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs