# %%
from ta.volatility import BollingerBands
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator
import pandas as pd
# Read CSV data
data = pd.read_csv('/Users/s106916/github/tradesformer/data/yahoo-EURUSD-2024-07-24-2024-09-21-5m.csv')
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
# %%
data
# %%
from sklearn.preprocessing import MinMaxScaler

# Select features
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'MACD', 'RSI', 'BB_High', 'BB_Low']
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Reset index
data = data.reset_index()
# %%
data
# %%
import numpy as np

sequence_length = 12  # Number of past observations to consider

def create_sequences(df, seq_length):
    sequences = []
    for i in range(len(df) - seq_length):
        seq = df.iloc[i:i+seq_length][features].values
        sequences.append(seq)
    return np.array(sequences)

sequences = create_sequences(data, sequence_length)

# %%
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)

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

    def forward(self, src):
        # Apply embedding layer and add positional encoding
        src = self.embedding(src) + self.positional_encoding

        # Pass through the transformer encoder
        output = self.transformer_encoder(src)

        # Apply layer normalization
        output = self.layer_norm(output)

        # Pass through the decoder layer
        output = self.decoder(output)

        # Check for NaN or Inf values for debugging
        if torch.isnan(output).any() or torch.isinf(output).any():
            raise ValueError("Transformer output contains NaN or Inf values")

        # Return the output from the last time step
        return output[:, -1, :]


# %%
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import torch.nn.functional as F

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=64)
        self.sequence_length = sequence_length
        num_features = observation_space.shape[1]  # Should be 10 in this case

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
            raise ValueError("Invalid values in transformer output")
        return x

# Update policy kwargs to use the custom extractor
policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(),
    net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    activation_fn=nn.ReLU
)

# %%
class ForexTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data, sequence_length, profit_target_ratio=0.01, stop_loss_ratio=0.005):
        super(ForexTradingEnv, self).__init__()
        self.data = data
        self.sequence_length = sequence_length
        self.max_steps = len(self.data) - self.sequence_length - 1
        self.current_step = 0

        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.positions = []
        self.total_profit = 0.0

        self.profit_target_ratio = profit_target_ratio
        self.stop_loss_ratio = stop_loss_ratio

        self.action_space = spaces.Discrete(3)

        obs_shape = (self.sequence_length, len(features))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.positions = []
        self.total_profit = 0.0

        self.current_step = np.random.randint(self.sequence_length, self.max_steps)

        observation = self._next_observation()
        info = {}
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

    def _calculate_reward(self, position, current_price):
        entry_price = position['entry_price']
        direction = position['direction']
        profit_target_price = entry_price * (1 + self.profit_target_ratio) if direction == 'buy' else entry_price * (1 - self.profit_target_ratio)
        stop_loss_price = entry_price * (1 - self.stop_loss_ratio) if direction == 'buy' else entry_price * (1 + self.stop_loss_ratio)

        should_close = False
        reward = 0.0

        # Check for profit target hit
        if (direction == 'buy' and current_price >= profit_target_price) or (direction == 'sell' and current_price <= profit_target_price):
            profit = abs(current_price - entry_price) * position['size']
            reward = profit  # Positive reward
            should_close = True

        # Check for stop-loss hit
        elif (direction == 'buy' and current_price <= stop_loss_price) or (direction == 'sell' and current_price >= stop_loss_price):
            loss = abs(current_price - entry_price) * position['size']
            reward = -loss  # Negative reward
            should_close = True

        return reward, should_close

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']

        # Execute action
        if action == 1:  # Buy
            if self.balance > 0:
                position_size = self.balance / current_price
                self.positions.append({
                    'direction': 'buy',
                    'entry_price': current_price,
                    'size': position_size
                })
                self.balance = 0
        elif action == 2:  # Sell
            if self.balance > 0:
                position_size = self.balance / current_price
                self.positions.append({
                    'direction': 'sell',
                    'entry_price': current_price,
                    'size': position_size
                })
                self.balance = 0

        # Initialize reward
        reward = 0.0

        # Update positions and calculate rewards
        positions_to_close = []
        for position in self.positions:
            position_reward, should_close = self._calculate_reward(position, current_price)
            reward += position_reward

            if should_close:
                # Close position
                if position['direction'] == 'buy':
                    self.balance += position['size'] * current_price
                else:
                    self.balance += position['size'] * (2 * position['entry_price'] - current_price)
                positions_to_close.append(position)

        # Remove closed positions
        for position in positions_to_close:
            self.positions.remove(position)

        # Update net worth
        self.net_worth = self.balance + sum(
            position['size'] * current_price if position['direction'] == 'buy' else position['size'] * (2 * position['entry_price'] - current_price)
            for position in self.positions
        )

        # Move to the next time step
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= self.max_steps

        # Get next observation
        obs = self._next_observation()

        # Additional info
        info = {}
        truncated = False
        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance:.2f}')
        print(f'Positions: {len(self.positions)}')
        print(f'Net Worth: {self.net_worth:.2f}')
        print(f'Profit: {profit:.2f}')

# %%
env = ForexTradingEnv(data, sequence_length)
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    policy_kwargs=policy_kwargs,
    learning_rate=1e-4,  # Reduced learning rate
    max_grad_norm=0.5    # Gradient clipping
)

# Train the agent
model.learn(total_timesteps=100000)

# Evaluate the agent
observation, info = env.reset()
done = False

while not done:
    action, _states = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()

# Save the model
model.save('ppo_forex_transformer')

# %%
