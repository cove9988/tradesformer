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

class TimeSeriesTransformer(nn.Module):
    """
    A Transformer-based model for time series data.
    This class projects input features to an embedding, adds positional
    encodings, and then processes the inputs using a Transformer encoder.
    Finally, a decoder layer is used to produce the output.
    Args:
        input_size (int): Number of features in the input time series data.
        embed_dim (int): Dimensionality of the learned embedding space.
        num_heads (int): Number of attention heads in each Transformer layer.
        num_layers (int): Number of Transformer encoder layers.
        sequence_length (int): Length of the input sequences (time steps).
        dropout (float, optional): Dropout probability to apply in the
            Transformer encoder layers. Defaults to 0.1.
    Attributes:
        model_type (str): Identifier for the model type ('Transformer').
        embedding (nn.Linear): Linear layer for input feature embedding.
        positional_encoding (torch.nn.Parameter): Parameter storing the
            positional encodings used to retain temporal information.
        transformer_encoder (nn.TransformerEncoder): Stack of Transformer
            encoder layers with optional final LayerNorm.
        decoder (nn.Linear): Linear layer used to produce the final output
            dimensions.
    Forward Inputs:
        src (torch.Tensor): Input tensor of shape (batch_size, sequence_length,
            input_size).
    Forward Returns:
        torch.Tensor: Output tensor of shape (batch_size, embed_dim) from the
            last time step.
    Raises:
        ValueError: If the model output contains NaN or Inf values, indicating
            numerical instability.
    """
    
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
    """
    A custom feature extractor that normalizes input observations and processes them
    using a transformer-based architecture for dimensionality reduction and enhanced
    feature representation.
    Parameters:
        observation_space (gym.spaces.Box): Defines the shape and limits of input data.
        sequence_length (int): The length of the time series to be processed.
    Attributes:
        layernorm_before (nn.LayerNorm): Normalizes input data to improve training stability.
        transformer (TimeSeriesTransformer): Processes normalized input sequences and extracts features.
    Methods:
        forward(observations):
            Applies layer normalization to the incoming observations, then passes them
            through the transformer. Raises a ValueError if invalid values (NaNs or inf)
            are detected in the output.
    """
    
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
    """
    A Gym-compatible environment for simulating a Forex trading scenario. It uses historical data and
    reinforcement learning concepts to place Buy or Sell orders while handling actions such as holding,
    stop-loss, profit-taking, and transaction fees.
    Attributes:
        file (str): Path to the CSV file containing historical market data.
        cf (object): Configuration manager or utility to retrieve environment parameters.
        asset (str): The name of the Forex pair or asset being traded.
        features (list): List of feature columns to be included in the observation space.
        sequence_length (int, optional): Number of time steps in each observation window. Defaults to 24.
        logger_show (bool, optional): If True, logs will be shown for each step. Defaults to False.
        save_plot (bool, optional): If True, trading charts will be saved. Defaults to False.
    Methods:
        reset(seed=None, options=None):
            Resets the environment to an initial state, returning the first observation and info.
        step(action):
            Executes a single time-step of the environment's dynamics. Takes an action, computes reward,
            and returns (observation, reward, done, truncated, info).
        render(mode='human', title=None, **kwargs):
            Provides different render modes: 'human' for console output, 'file' to log info,
            'graph' to generate a chart, and 'both' to combine logging and charting.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self, file, cf, asset, features, sequence_length = 24, logger_show = False, save_plot = False):
        super(ForexTradingEnv, self).__init__()
        self._initialize_parameters(file, cf, asset, features, sequence_length, logger_show, save_plot)
        self._initialize_spaces()
        self.reset()

    def _initialize_parameters(self, file, cf, asset, features, sequence_length, logger_show, save_plot):
        self.csv_file = file
        self.cf = cf
        self.data = pd.read_csv(file)
        self.features = features
        self.logger_show = logger_show
        self.save_plot = save_plot
        self.sequence_length = sequence_length 
        self.max_steps = len(self.data) - self.sequence_length - 1
        self.balance_initial = self.cf.env_parameters("balance")
        self.good_position_encourage = self.cf.env_parameters("good_position_encourage")
        self.consistency_reward = self.cf.env_parameters("consistency_reward")
        self.shaping_reward = self.cf.env_parameters("shaping_reward")
        self.symbol_col = asset
        self.stop_loss = self.cf.symbol(self.symbol_col, "stop_loss_max")
        self.profit_taken = self.cf.symbol(self.symbol_col, "profit_taken_max")
        self.point = self.cf.symbol(self.symbol_col, "point")
        self.transaction_fee = self.cf.symbol(self.symbol_col, "transaction_fee")
        self.over_night_penalty = self.cf.symbol(self.symbol_col, "over_night_penalty")
        self.max_current_holding = self.cf.symbol(self.symbol_col, "max_current_holding")

    def _initialize_spaces(self):
        self.action_space = spaces.Discrete(3)
        obs_shape = (self.sequence_length, len(self.features))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        self.ticket_id = 0
        self.ttl_rewards = 0
        self.current_step = 0
        self.balance = self.balance_initial
        self.positions = []
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
        """_summary_
        dynamic PT and SL by adding the reward for this step. if we have correct direction, both PT and SL will left up, otherwise, it will
        drop down
        so every step, need to recalculate the PT and SL
        PT = PT + reward, SL = SL + abs(reward)
        """
        _o, _c, _h, _l,_t,_day = self.data.iloc[self.current_step][["open","close","high","low","time","day"]]
        _msg =[]
        entry_price = position['ActionPrice']
        direction = position['Type']
        profit_target_price = entry_price + position['PT']/self.point if direction == 'Buy' else entry_price - position['PT']/self.point
        stop_loss_price = entry_price + position['SL']/self.point if direction == 'Buy' else entry_price - position['SL']/self.point
        closed = False
        close_position_reward = 0.0
        good_position_reward = 0.0
        # Check for stop-loss hit
        if (direction == 'Buy' and _l <= stop_loss_price) or (direction == 'Sell' and _h >= stop_loss_price):
            # loss = abs(current_price - entry_price) * position['size']
            close_position_reward = position['SL']  # Negative close_position_reward
            position["CloseTime"] = _t
            position["ClosePrice"] = stop_loss_price
            position["Status"] = 1
            position["CloseStep"] = self.current_step
            position["pips"] += close_position_reward
            position["DeltaStep"] = self.current_step - position["ActionStep"]
            self.balance += 100 + position["pips"] #return deposit
            closed = True
            _msg.append(f'Step:{self.current_step} Tkt:{position["Ticket"]}: Rwd:{position["pips"]}, SL:{position["SL"]}, DeltaStep:{position["DeltaStep"]}')
            close_position_reward = position['SL']  # Negative close_position_reward
        elif (direction == 'Buy' and _h >= profit_target_price) or (direction == 'Sell' and _l <= profit_target_price):
            # profit = abs(current_price - entry_price) * position['size']
            close_position_reward =  position['PT'] # Positive close_position_reward
            position["CloseTime"] = _t
            position["pips"] += close_position_reward
            position["Status"] = 1
            position["CloseStep"] = self.current_step
            position["pips"] += close_position_reward
            position["DeltaStep"] = self.current_step - position["ActionStep"]
            self.balance += 100 + position["pips"]
            closed = True
            _msg.append(f'Step:{self.current_step} Tkt:{position["Ticket"]}: Rwd:{position["pips"]}, PT:{position["PT"]}, DeltaStep:{position["DeltaStep"]}')
            close_position_reward =  position['PT'] # Positive close_position_reward
        else:
            if self.current_step + 5 + self.sequence_length >= len(self.data): # close all position 5 steps before all end.
                close_position_reward = (_c - position["ActionPrice"] if direction == 'Buy' else position["ActionPrice"] - _c)* self.point
                position["CloseTime"] = _t
                position["ClosePrice"] = _c
                position["Status"] = 2
                position["CloseStep"] = self.current_step
                position["pips"] += close_position_reward
                position["DeltaStep"] = self.current_step - position["ActionStep"]
                _msg.append(f'Step:{self.current_step} Tkt:{position["Ticket"]}: Rwd:{position["pips"]}, Cls:End, DeltaStep:{position["DeltaStep"]}')
                self.balance += 100 + position["pips"]
                closed = True
                close_position_reward = (_c - position["ActionPrice"] if direction == 'Buy' else position["ActionPrice"] - _c)* self.point
                delta = _c - position["ActionPrice"]
                if direction == "Buy":
                    good_position_reward = self.good_position_encourage if delta >=0 else -self.good_position_encourage
                elif direction == "Sell":
                    good_position_reward = -self.good_position_encourage if delta >=0 else self.good_position_encourage

                position["PT"] += good_position_reward 
                position["SL"] = position["SL"] + good_position_reward if position["SL"] >= self.stop_loss else self.stop_loss
                    
                position["Reward"] += good_position_reward
                _msg.append(f'Step:{self.current_step} Tkt:{position["Ticket"]}: NO_Close, PT:{position["PT"]}, SL:{position["SL"]} Rwd:{position["Reward"]}')    
        return close_position_reward + good_position_reward, closed, _msg

    def step(self, action):
        _o, _c, _h, _l,_t,_day = self.data.iloc[self.current_step][["open","close","high","low","time","day"]]
        reward = 0.0
        position_reward = 0
        stability_reward = 0
        action_hold_reward = 0
        
        _msg =[]

        open_positon = 0
        for position in self.positions:
            if position['Status'] == 0:
                position_reward, closed,_msg = self._calculate_reward(position)
                if not closed: open_positon += 1
                reward += position_reward
        # Execute action
        _action, stability_reward = self.action_aggregator.add_action(action) 
        if open_positon < self.max_current_holding: #only check if need to open
            reward += stability_reward
        # logger.info(f'Step:{self.current_step}: action: {action}, real: {_action} stability reward:{stability_reward} ')
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
            _msg.append(f'Step:{self.current_step} Tkt:{position["Ticket"]} {position["Type"]} Rwd:{position["pips"]} SL:{position["SL"]} PT:{position["PT"]}')
        elif open_positon < self.max_current_holding and action == 0:
            action_hold_reward = -1 # no open any position, encourage open position
        else:
            action_hold_reward = 0
        
        reward += action_hold_reward 
            
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
            _m = f'--- Position:{len(self.positions)}/Buy:{buy} TtlRwds: {self.ttl_rewards} Balance: {self.balance} step {self.current_step }'
            logger.info (_m)
            _msg.append(_m)
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
            p = TradingChart(self.csv_file, self.positions, self.save_plot)
            p.plot()
        elif mode == 'both':
            log_header = True
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
            
            p = TradingChart(self.csv_file, self.positions, self.save_plot)
            p.plot()
