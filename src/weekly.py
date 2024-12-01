# %%
import pandas as pd
import numpy as np
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta, datetime
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from ta.volatility import BollingerBands
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator
from log_render import render_to_file
from plot_chart import TradingChart

CLOSE_REASSON = ['PT', 'SL', 'END']
CURRENCY_PAIR = 'EURUSD'
POSITION_SIZE = 0.1
INITIAL_BALANCE = 10000.0
OPEN_POSITIONS_COST_PIPS = 3.00
PROFIT_TARGET_PIPS=80.00
STOP_LOSS_PIPS=40.00
LOSS_HIT_FIRST = True
SEQUENCE_LENGTH = 12
TOTAL_TIMESTEPS_PER_WEEK = 20000
MODEL_NAME = './model/ppo_forex_transformer_final.zip'
pip_factor = 10000.0
# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)

# -------------------------
# Data Loading and Preprocessing
# -------------------------

# Load your data
data = pd.read_csv('yahoo-EURUSD-2024-07-24-2024-09-21-5m.csv', index_col='Datetime', parse_dates=True)
# Calculate indicators
data['SMA'] = SMAIndicator(data['Close'], window=14).sma_indicator()
data['MACD'] = MACD(data['Close']).macd()
data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()

# Bollinger Bands
bb_indicator = BollingerBands(data['Close'])
data['BB_High'] = bb_indicator.bollinger_hband()
data['BB_Low'] = bb_indicator.bollinger_lband()

# Drop NaN values
data = data.dropna()
# Add technical indicators (assuming you've already done this)
# ... (Add your technical indicators here) ...

# Drop NaN values
data = data.dropna()

# Normalize features
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'MACD', 'RSI', 'BB_High', 'BB_Low']
# scaler = MinMaxScaler()
# data[features] = scaler.fit_transform(data[features])

# Add week number and year columns
data['Week_Number'] = data.index.isocalendar().week
data['Year'] = data.index.isocalendar().year

# Group data by weeks
grouped_weeks = data.groupby(['Year', 'Week_Number'])

# Create a list of weekly DataFrames
weekly_data_list = [group for _, group in grouped_weeks]# %%
# Save each weekly DataFrame to a CSV file
for i, weekly_data in enumerate(weekly_data_list):
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    weekly_data.to_csv(f'./data/weekly_data_week_{i + 1}.csv')
# %%
def calculate_pips(open_price, close_price, pair, position_type):
    """
    Calculate the number of pips gained or lost in a forex trade.

    Args:
        open_price (float): The price at which the position was opened.
        close_price (float): The price at which the position was closed.
        pair (str): The currency pair traded (e.g., 'EUR/USD', 'USD/JPY').
        position_type (str): 'buy' or 'sell'.

    Returns:
        float: The number of pips gained (positive) or lost (negative).
    """
    # Determine the pip factor based on the currency pair
    if 'JPY' in pair:
        pip_factor = 100.0
    else:
        pip_factor = 10000.0

    # Calculate the pip difference
    if position_type.lower() == 'b':
        pip_difference = (close_price - open_price) * pip_factor
    elif position_type.lower() == 's':
        pip_difference = (open_price - close_price) * pip_factor
    else:
        raise ValueError("position_type must be 'buy' or 'sell'.")

    return pip_difference

# Example usage:

# # For a EUR/USD buy trade
# open_price = 1.1050
# close_price = 1.1075
# pair = 'EUR/USD'
# position_type = 'buy'

# pips = calculate_pips(open_price, close_price, pair, position_type)
# print(f'Pips gained: {pips:.2f}')

# # For a USD/JPY sell trade
# open_price = 110.00
# close_price = 109.50
# pair = 'USD/JPY'
# position_type = 'sell'

# pips = calculate_pips(open_price, close_price, pair, position_type)
# print(f'Pips gained: {pips:.2f}')


# -------------------------
# Define the Environment
# -------------------------

class ForexTradingEnv(gym.Env):
    metadata = {'render.modes': ['graph', 'human', 'file']}

    def __init__(self, data, sequence_length, profit_target_pips=PROFIT_TARGET_PIPS, stop_loss_pips=STOP_LOSS_PIPS,render_mode=None):
        super(ForexTradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.max_steps = len(self.data) - self.sequence_length - 1
        self.current_step = 0

        self.initial_balance = INITIAL_BALANCE
        self.balance = self.initial_balance
        self.positions = []
        self.pips = 0
        self.tranactions_id = 1
        self.profit_target_pips = profit_target_pips
        self.stop_loss_pips = stop_loss_pips
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.log_header = True
        self.log_filename = './data/log/' + datetime.now().strftime('%Y%m%d%H%M%S') + '.csv'
        self.visualization = False
        obs_shape = (self.sequence_length, len(features))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

    def set_new_data(self, new_data):
        self.data = new_data.reset_index(drop=True)
        self.max_steps = len(self.data) - self.sequence_length - 1
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.positions = []
        self.total_profit = 0.0
        self.log_header = True
        self.visualization = False
        self.current_step = np.random.randint(self.sequence_length, self.max_steps)

        observation = self._next_observation()
        info = {'step': self.current_step,
                'balance': self.balance,
                'net_worth': self.net_worth,
                'total_profit': self.total_profit,
                'positions': self.positions,
                'profit_target_pips': self.profit_target_pips,
                'stop_loss_pips': self.stop_loss_pips,
                }
        
        return observation, info

    def _next_observation(self):
        obs = self.data.iloc[
            self.current_step - self.sequence_length : self.current_step
        ][features].values
        # Convert to float32 tensor
        obs = torch.tensor(obs, dtype=torch.float32)
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            raise ValueError(f"Invalid observation at step {self.current_step}")
        return obs

    def _calculate_reward(self, position, current_price,high_price,low_price):
        open_price = position['open_price']
        direction = position['direction']
        profit_target_price = open_price * (1 + self.profit_target_pips/pip_factor) if direction == 'b' else open_price * (1 - self.profit_target_pips/pip_factor)
        stop_loss_price = open_price * (1 - self.stop_loss_pips/pip_factor) if direction == 'b' else open_price * (1 + self.stop_loss_pips/pip_factor)

        reward = 0.0
        self.pips = 0.0
        # Check for profit target hit
        # if LOSS_HIT_FIRST: to it later
        # Check for stop-loss hit
        if (direction == 'b' and low_price <= stop_loss_price) or (direction == 's' and high_price >= stop_loss_price):
            # loss = abs(stop_loss_price - open_price) * position['size']
            self.pips = calculate_pips(open_price=open_price, close_price=stop_loss_price, pair=position['currency_pair'], position_type=direction)
            position['close_price'] = stop_loss_price
            position['close_step'] = self.current_step
            # position['close_time'] = self.data.iloc[self.current_step]['Datetime']
            position['close_reason'] = 'SL'
            position['pips'] = self.pips
            position['duration'] = self.current_step - position['open_step']

        
        elif (direction == 'b' and high_price >= profit_target_price) or (direction == 's' and low_price <= profit_target_price):
            # profit = abs(profit_target_price - open_price) * position['size']
            self.pips = calculate_pips(open_price=open_price, close_price=profit_target_price, pair=position['currency_pair'], position_type=direction)
            position['close_price'] = profit_target_price
            position['close_step'] = self.current_step
            # position['close_time'] = self.data.iloc[self.current_step]['Datetime']
            position['close_reason'] = 'PT'
            position['pips'] = self.pips
            position['duration'] = self.current_step - position['open_step']

        else: # if end of data, close it
            if self.current_step == self.max_steps - 1:
                if direction == 'b':
                    # profit = (current_price - open_price) * position['size']
                    self.pips = calculate_pips(open_price=open_price, close_price=current_price, pair=position['currency_pair'], position_type=direction)
                else:
                    # profit = (open_price - current_price) * position['size']
                    self.pips = calculate_pips(open_price=open_price, close_price=current_price, pair=position['currency_pair'], position_type=direction)
                position['close_price'] = current_price
                position['close_step'] = self.current_step
                # position['close_time'] = self.data.iloc[self.current_step]['Datetime']
                position['close_reason'] = 'END'
                position['pips'] = self.pips
                position['duration'] = self.current_step - position['open_step']

        reward = self.pips
        return reward

    def step(self, action):
        low_price = self.data.iloc[self.current_step]['Low']
        high_price = self.data.iloc[self.current_step]['High']
        current_price = self.data.iloc[self.current_step]['Close']

        # Initialize reward
        reward = 0.0

        # Update positions and calculate rewards
        for position in self.positions:
            if position['close_reason'] == '':
                reward = self._calculate_reward(position, current_price,high_price,low_price)

        # Execute action
        if action == 1:  # Buy
            if self.balance > 0:
                position_size = POSITION_SIZE #self.balance / current_price
                self.positions.append({
                    'tranactions_id': self.tranactions_id,
                    'direction': 'b',
                    'size': position_size,
                    'open_price': current_price,
                    'open_step': self.current_step,
                    # 'open_time': self.data.iloc[self.current_step]['Datetime'],
                    'close_price': 0,
                    'close_step': -1,
                    # 'close_time': '',
                    'close_reason': '',
                    'pips': 0,
                    'currency_pair': CURRENCY_PAIR,
                    'duration':0,
                    'maxDD':0.00                    
                })
                reward -= OPEN_POSITIONS_COST_PIPS
                self.tranactions_id += 1
        elif action == 2:  # Sell
            if self.balance > 0:
                position_size = POSITION_SIZE #self.balance / current_price
                self.positions.append({
                    'tranactions_id': self.tranactions_id,
                    'direction': 's',
                    'size': position_size,
                    'open_price': current_price,
                    'open_step': self.current_step,
                    # 'open_time': self.data.iloc[self.current_step]['Datetime'],
                    'close_price': 0,
                    'close_step': -1,
                    # 'close_time': '',
                    'close_reason': '',
                    'pips': 0,
                    'currency_pair': CURRENCY_PAIR,
                    'duration':0,
                    'maxDD':0.00
                })
                reward -= OPEN_POSITIONS_COST_PIPS
                self.tranactions_id += 1
        if np.isnan(reward) or np.isinf(reward):
            print(f"Invalid reward at step {self.current_step}")
            raise ValueError("Reward contains NaN or Inf values")
        # Move to the next time step
        self.current_step += 1
        self.balance += reward

        # Determine if the episode is terminated or truncated
        terminated = False
        if self.current_step >= self.max_steps or self.balance <= 0:
            terminated = True
            self.visualization = True
        truncated = False  # Update if you have truncation conditions

        # Get next observation
        observation = self._next_observation()

        # Additional info
        info = {'step': self.current_step,
                'balance': self.balance,
                'net_worth': self.net_worth,
                'total_profit': self.total_profit,
                'positions': self.positions,
                'profit_target_pips': self.profit_target_pips,
                'stop_loss_pips': self.stop_loss_pips,
                }

        return observation, reward, terminated, truncated, info

    # def render(self, mode='human'):
    #     p =""
    #     for position in self.positions:
    #         p +=(f"[{position['direction']} {position['open_price']:8.4f} {position['open_step']}]") 
    #     print(f'Step: {self.current_step:07} Pips: {self.pips:+7.2f} Balance: {self.balance:8.2f} Positions: {len(self.positions):02} {p}')
    #     # print(f'Net Worth: {self.net_worth:.2f}')
    #     # print(f'Profit: {profit:.2f}')

    def render(self, mode='human', title=None, **kwargs):
        # Render the environment to the screen
        if mode in ('human', 'file'):
            printout = False
            if mode == 'human':
                printout = True
            pm = {
                "step": self.current_step,
                "log_header": self.log_header,
                "log_filename": self.log_filename,
                "printout": printout,
                "balance": self.balance,
                "balance_initial": self.initial_balance,
                "positions": self.positions, # [p for p in self.positions if p['close_step'] == self.current_step],
                "done_information": "",
            }
            render_to_file(**pm)
            if self.log_header: self.log_header = False
        elif mode == 'graph' and self.visualization:
            print('plotting...')
            p = TradingChart(self.df, self.positions)
            p.plot()
# -------------------------
# Define the Transformer Model
# -------------------------

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.embed_dim = embed_dim

        # Embedding layer to project input features to embed_dim dimensions
        self.embedding = nn.Linear(input_size, embed_dim)

        # Positional encoding parameter
        self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_length, embed_dim))

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Decoder layer to produce final output
        self.decoder = nn.Linear(embed_dim, embed_dim)
        # Add dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Apply embedding layer and add positional encoding
        src = self.embedding(src) + self.positional_encoding

        # Pass through the transformer encoder
        output = self.transformer_encoder(src)
        output = self.dropout(output)
        # Apply layer normalization
        output = self.layer_norm(output)

        # Pass through the decoder layer
        output = self.decoder(output)

        # Check for NaN or Inf values for debugging
        if torch.isnan(output).any() or torch.isinf(output).any():
            raise ValueError("Transformer output contains NaN or Inf values")
        
        # Return the output from the last time step
        return output[:, -1, :]

# -------------------------
# Custom Feature Extractor
# -------------------------

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=64)
        self.sequence_length = observation_space.shape[0]
        num_features = observation_space.shape[1]

        # Ensure that embed_dim is divisible by num_heads
        embed_dim = 64
        num_heads = 2

        self.transformer = TimeSeriesTransformer(
            input_size=num_features,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=2
        )

    def forward(self, observations):
        x = self.transformer(observations)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Invalid values in transformer output")
            raise ValueError("Transformer output contains NaN or Inf values")
        return x

# -------------------------
# Training Loop
# -------------------------

# Define policy kwargs
policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(),
    net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    activation_fn=nn.ReLU
)

# Initialize the environment and the model
sequence_length = SEQUENCE_LENGTH  # Number of past observations to consider
env = ForexTradingEnv(data=weekly_data_list[0], sequence_length=sequence_length,render_mode='human')
env = DummyVecEnv([lambda: env])

model = PPO(
    'MlpPolicy', 
    env, 
    verbose=1, 
    policy_kwargs=policy_kwargs, 
    learning_rate=1e-5,  # Consider lowering it further if needed
    max_grad_norm=0.5,    # Gradient clipping
    tensorboard_log='./ppo_forex_tensorboard/')

# Training parameters
total_timesteps_per_week = TOTAL_TIMESTEPS_PER_WEEK  # Adjust based on your computational resources
# %%
# Loop over each week's data
start_time = datetime.now()
print(f"Training started at: {start_time}")

# Loop over each week's data
for week_index, weekly_data in enumerate(weekly_data_list):
    print(f"\n--- Training on Week {week_index + 1} ---\n")

    # Update the environment with the new week's data
    env.env_method('set_new_data', weekly_data)

    # Reset the environment
    obs = env.reset()

    # Train the model on the current week's data
    model.learn(total_timesteps=total_timesteps_per_week)

    # Optionally, save the model after each week
    model.save(f'./model/ppo_forex_transformer_week_{week_index + 1}')
    # model.policy.features_extractor.transformer.save_model(f'transformer_model_week_{week_index + 1}.pth')
    print(f"Training {week_index} end at: {datetime.now()}")
# After training on all weeks, save the final model
model.save('./model/ppo_forex_transformer_final')
# model.policy.features_extractor.transformer.save_model('transformer_model_final.pth')
# %%
# -------------------------
# Evaluation
# -------------------------
env = ForexTradingEnv(data=weekly_data_list[0], sequence_length=sequence_length,render_mode='human')
env = DummyVecEnv([lambda: env])
model_name = MODEL_NAME
if model_name:
    model = PPO.load(model_name, env=env)
# Evaluate the agent on the last week's data
env.env_method('set_new_data', weekly_data_list[2])
# env.envs[0].env.set_new_data(weekly_data_list[-1])
obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, info = env.step(action)
    done = terminated
    env.render(mode = 'human')


# %%
