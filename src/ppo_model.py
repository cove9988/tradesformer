# %%
import os
import logging
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.callbacks import LearningRateSchedule
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import torch.nn.functional as F
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import datetime
from src.util.plot_chart import TradingChart
from src.util.log_render import render_to_file
from src.util.action_aggregation import ActionAggregator

logger = logging.getLogger(__name__)
# def linear_schedule(initial_value: float):
#     def func(progress_remaining: float) -> float:
#         return progress_remaining * initial_value
#     return func
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def load_data(csv_file):
    data = pd.read_csv(csv_file)


    sequence_length = len(features)  # Number of past observations to consider
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])

    # Check for NaN or Inf values after scaling
    if data[features].isnull().values.any() or np.isinf(data[features].values).any():
        logger.error("Data contains NaN or Inf values after scaling")
        raise ValueError("Data contains NaN or Inf values after scaling")

    # Reset index
    data = data.reset_index()
    logger.info("Data loaded and preprocessed successfully")

    def create_sequences(df, seq_length):
        logger.info("Creating sequences...")
        sequences = []
        for i in range(len(df) - seq_length):
            seq = df.iloc[i:i+seq_length][features].values
            sequences.append(seq)
        logger.info("Sequences created successfully")
        return np.array(sequences)

    sequences = create_sequences(data, sequence_length)
    return data

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads, num_layers,sequence_length, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.embed_dim = embed_dim

        # Embedding layer to project input features to embed_dim dimensions
        self.embedding = nn.Linear(input_size, embed_dim).to(device)

        # Positional encoding parameter
        self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_length, embed_dim).to(device))

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            norm_first=True  # Apply LayerNorm before attention and feedforward
        ).to(device)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim).to(device) # Add LayerNorm at the end of the encoder
        )

        # Decoder layer to produce final output
        self.decoder = nn.Linear(embed_dim, embed_dim).to(device)

    def forward(self, src):
        # Apply embedding layer and add positional encoding
        src = self.embedding(src) + self.positional_encoding

        # Pass through the transformer encoder
        output = self.transformer_encoder(src)

        # Pass through the decoder layer
        output = self.decoder(output)

        # Check for NaN or Inf values for debugging
        if torch.isnan(output).any() or torch.isinf(output).any():
            logger.error("Transformer output contains NaN or Inf values")
            raise ValueError("Transformer output contains NaN or Inf values")

        # Return the output from the last time step
        return output[:, -1, :]

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, sequence_length):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=64)
        num_features = observation_space.shape[1]  # Should be 10 in this case

        # Ensure that embed_dim is divisible by num_heads
        embed_dim = 64
        num_heads = 2

        self.layernorm_before = nn.LayerNorm(num_features).to(device) # Added Layer Normalization before transformer

        self.transformer = TimeSeriesTransformer(
            input_size=num_features,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=2,
            sequence_length =sequence_length
        ).to(device)

    def forward(self, observations):
        # Apply layer normalization
        normalized_observations = self.layernorm_before(observations.float().to(device)) # Ensure float type

        x = self.transformer(normalized_observations)
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("Invalid values in transformer output")
            raise ValueError("Invalid values in transformer output")
        return x

# %%
class ForexTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, file, cf, asset, features, sequence_length = 24, logger_show = False):
        super(ForexTradingEnv, self).__init__()
        self.csv_file = file
        self.cf = cf
        self.data = pd.read_csv(file)
        self.features = features
        self.logger_show = logger_show
        self.sequence_length = sequence_length 
        self.max_steps = len(self.data) - self.sequence_length - 1
        self.balance_initial = self.cf.env_parameters("balance")
        self.good_position_encourage=self.cf.env_parameters("good_position_encourage")
        self.consistency_reward =self.cf.env_parameters("consistency_reward")
        self.shaping_reward = self.cf.env_parameters("shaping_reward")
        self.symbol_col = asset
        self.stop_loss = self.cf.symbol(self.symbol_col, "stop_loss_max")
        self.profit_taken = self.cf.symbol(self.symbol_col, "profit_taken_max")
        self.point =self.cf.symbol(self.symbol_col, "point")
        self.transaction_fee=self.cf.symbol(self.symbol_col, "transaction_fee")
        self.over_night_penalty =self.cf.symbol(self.symbol_col, "over_night_penalty")
        self.max_current_holding =self.cf.symbol(self.symbol_col, "max_current_holding")
        self.action_space = spaces.Discrete(3)
        # self.backward_window = self.self.self.cf.env_parameters("backward_window")
        # need to reset of following
        self.current_step = 0
        self.ticket_id = 0
        self.balance = self.balance_initial
        self.positions = []
        self.ticket_id = 0
        self.ttl_rewards = 0
        self.action_aggregator =ActionAggregator()
        # self.reward_calculator = RewardCalculator(
        #     self.data, cf, self.shaping_reward, self.stop_loss, self.profit_taken, self.backward_window
        # )
        # self.transaction_manager = TransactionManager(
        #     cf, self.balance_initial, cf.env_parameters("symbol_col"), self.stop_loss, self.profit_taken
        # )

        obs_shape = (self.sequence_length, len(features))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)

        self.current_step = 0
        self.ticket_id = 0
        self.balance = self.balance_initial
        self.positions = []
        self.ticket_id = 0
        self.ttl_rewards = 0.0
        self.action_aggregator =ActionAggregator()
        
        # self.current_step = np.random.randint(self.sequence_length, self.max_steps)
        self.current_step = self.sequence_length
        logger.info(f"--- Environment reset. Starting at step {self.current_step} --total rewards: {self.ttl_rewards}")

        observation = self._next_observation()
        info = {}
        return observation, info

    def _next_observation(self):
        obs = self.data.iloc[
            self.current_step - self.sequence_length : self.current_step
        ][self.features].values
        # Convert to float32 tensor
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            logger.error(f"Invalid observation at step {self.current_step}")
            raise ValueError(f"Invalid observation at step {self.current_step}")
        return obs.cpu().numpy() #obs

    def _calculate_reward(self, position):
        '''
        dynamic PT and SL by adding the reward for this step. if we have correct direction, both PT and SL will left up, otherwise, it will
        drop down
        so every step, need to recalculate the PT and SL
        PT = PT + reward, SL = SL + abs(reward)
        '''
        _o, _c, _h, _l,_t,_day = self.data.iloc[self.current_step][["open","close","high","low","time","day"]]
        _msg =[]
        entry_price = position['ActionPrice']
        direction = position['Type']
        profit_target_price = entry_price + position['PT']/self.point if direction == 'Buy' else entry_price - position['PT']/self.point
        stop_loss_price = entry_price + position['SL']/self.point if direction == 'Buy' else entry_price - position['SL']/self.point
        closed = False
        close_poistion_reward = 0.0
        good_position_reward = 0.0
        # Check for stop-loss hit
        if (direction == 'Buy' and _l <= stop_loss_price) or (direction == 'Sell' and _h >= stop_loss_price):
            # loss = abs(current_price - entry_price) * position['size']
            close_poistion_reward = position['SL']  # Negative close_poistion_reward
            position["CloseTime"] = _t
            position["ClosePrice"] = stop_loss_price
            position["Status"] = 1
            position["CloseStep"] = self.current_step
            position["pips"] += close_poistion_reward
            position["DeltaStep"] = self.current_step - position["ActionStep"]
            self.balance += 100 + position["pips"] #return deposit
            closed = True
            _msg.append(f"Step:{self.current_step} Tkt:{position["Ticket"]}: Rwd:{position["pips"]}, SL:{position["SL"]}, DeltaStep:{position['DeltaStep']}")
        # Check for profit target hit
        elif (direction == 'Buy' and _h >= profit_target_price) or (direction == 'Sell' and _l <= profit_target_price):
            # profit = abs(current_price - entry_price) * position['size']
            close_poistion_reward =  position['PT'] # Positive close_poistion_reward
            position["CloseTime"] = _t
            position["ClosePrice"] = profit_target_price
            position["Status"] = 1
            position["CloseStep"] = self.current_step
            position["pips"] += close_poistion_reward
            position["DeltaStep"] = self.current_step - position["ActionStep"]
            self.balance += 100 + position["pips"]
            closed = True
            _msg.append(f"Step:{self.current_step} Tkt:{position["Ticket"]}: Rwd:{position["pips"]}, PT:{position["PT"]}, DeltaStep:{position['DeltaStep']}")

        else:
            if self.current_step + 5 + self.sequence_length >= len(self.data): # close all position 5 steps before all end.
                close_poistion_reward = (_c - position["ActionPrice"] if direction == 'Buy' else position["ActionPrice"] - _c)* self.point
                position["CloseTime"] = _t
                position["ClosePrice"] = _c
                position["Status"] = 2
                position["CloseStep"] = self.current_step
                position["pips"] += close_poistion_reward
                position["DeltaStep"] = self.current_step - position["ActionStep"]
                _msg.append(f"Step:{self.current_step} Tkt:{position["Ticket"]}: Rwd:{position["pips"]}, Cls:End, DeltaStep:{position['DeltaStep']}")
                self.balance += 100 + position["pips"]
                closed = True
            else:
                delta = _c - position["ActionPrice"]
                if direction == "Buy":
                    good_position_reward = self.good_position_encourage if delta >=0 else -self.good_position_encourage
                elif direction == "Sell":
                    good_position_reward = -self.good_position_encourage if delta >=0 else self.good_position_encourage

                position["PT"] += good_position_reward 
                position["SL"] = position["SL"] + good_position_reward if position["SL"] >= self.stop_loss else self.stop_loss
                    
                position["Reward"] += good_position_reward
                _msg.append(f"Step:{self.current_step} Tkt:{position["Ticket"]}: NO_Close, PT:{position['PT']}, SL:{position['SL']} Rwd:{position["Reward"]}")    
        return close_poistion_reward + good_position_reward, closed, _msg

    def step(self, action):
        _o, _c, _h, _l,_t,_day = self.data.iloc[self.current_step][["open","close","high","low","time","day"]]
        reward = 0.0
        position_reward = 0
        stability_reward = 0
        action_hold_reward = 0
        
        _msg =[]
        # Update positions and calculate rewards
        open_positon = 0
        for position in self.positions:
            if position['Status'] == 0:
                position_reward, closed,_msg = self._calculate_reward(position)
                if not closed: open_positon += 1
                reward += position_reward
        # Execute action
        _action, stability_reward = self.action_aggregator.add_action(action) 
        reward += stability_reward
        # logger.info(f"Step:{self.current_step}: action: {action}, real: {_action} stability reward:{stability_reward} ")
        if _action in (1, 2) and open_positon < self.max_current_holding :
            self.ticket_id += 1
            position = {
                "Ticket": self.ticket_id,
                "Symbol": self.symbol_col,
                "ActionTime": _t,
                "Type": "Buy" if _action ==1 else "Sell",
                "Lot": 1,
                "ActionPrice": _c,
                "SL": self.stop_loss,
                "PT": self.profit_taken,
                "MaxDD": 0,
                "Swap": 0.0,
                "CloseTime": "",
                "ClosePrice": 0.0,
                "Point": self.point,
                "Reward": self.transaction_fee,
                "DateDuration": _day,
                "Status": 0,
                "LimitStep": 0,
                "pips":self.transaction_fee,
                "ActionStep":self.current_step,
                "CloseStep":-1,
                "DeltaStep" : 0
            }
            self.positions.append(position)
            # do not use transaction_fee penalty  
            # reward = self.transaction_fee #open cost
            self.balance -= 100 # hold up, this will make sure model can not open a lot of
            _msg.append(f"Step:{self.current_step} Tkt:{position['Ticket']} {position['Type']} Rwd:{position['pips']} SL:{position['SL']} PT:{position['PT']}")
        else:
            action_hold_reward = -1
        
        reward += action_hold_reward # no open any position, encourage open position
            
        self.ttl_rewards += reward    
        # Move to the next time step
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= self.max_steps or self.balance <= 0

        # Get next observation
        obs = self._next_observation()
        _msg.append(f'---idle----step:{self.current_step}, RF:{action} Action:{_action} reward:{reward} total_rewards:{self.ttl_rewards} position_reward:{position_reward} stability_reward:{stability_reward} action_hold_reward:{action_hold_reward}')
        # Convert tensors to CPU for logging or NumPy conversion
        # obs_cpu = obs.cpu().numpy()
        if done:
            buy = 0
            for position in self.positions:
                if position["Type"] == "Buy":
                    buy +=1

            _msg.append(f'--- Position:{len(self.positions)}/Buy:{buy} TtlRwds: {self.ttl_rewards} Balance: {self.balance} step {self.current_step }')
        # Additional info
        if self.logger_show:
            for _m in _msg:
                logger.info(_m)
        info = {"info":_msg}
        truncated = False
        return obs, reward, done, truncated, info

    def render(self, mode='human', title=None, **kwargs):
        # Render the environment to the screen
        if mode in ('human', 'file'):
            log_header = True
            printout = False
            if mode == 'human':
                printout = True
            log_file = self.csv_file.replace("split/", "log/")
            pm = {
                "log_header": log_header,
                "log_filename": log_file,
                "printout": printout,
                "balance": self.balance,
                "balance_initial": self.balance_initial,
                "transaction_close_this_step": self.positions,
                "done_information": False
            }
            render_to_file(**pm)
            if log_header: log_header = False
        elif mode == 'graph' :
            p = TradingChart(self.csv_file, self.positions)
            p.plot()


