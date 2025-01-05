# %%
import logging
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import torch.nn.functional as F
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
# %%
data = pd.read_csv("/home/paulg/github/tradesformer/data/split/AUDUSD/weekly/AUDUSD_2022_1.csv")

features = ['open', 'high', 'low', 'close', 'vol', 'macd','boll_ub','boll_lb','rsi_30','dx_30','close_30_sma','close_60_sma']
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
# %%
def create_sequences(df, seq_length):
    logger.info("Creating sequences...")
    sequences = []
    for i in range(len(df) - seq_length):
        seq = df.iloc[i:i+seq_length][features].values
        sequences.append(seq)
    logger.info("Sequences created successfully")
    return np.array(sequences)

sequences = create_sequences(data, sequence_length)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads, num_layers, dropout=0.1):
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
    def __init__(self, observation_space: gym.spaces.Box):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=64)
        self.sequence_length = sequence_length
        num_features = observation_space.shape[1]  # Should be 10 in this case

        # Ensure that embed_dim is divisible by num_heads
        embed_dim = 64
        num_heads = 2

        self.layernorm_before = nn.LayerNorm(num_features).to(device) # Added Layer Normalization before transformer

        self.transformer = TimeSeriesTransformer(
            input_size=num_features,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=2
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
policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(),
    net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    activation_fn=nn.ReLU
)

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
        logger.info(f"Environment reset. Starting at step {self.current_step}")

        observation = self._next_observation()
        info = {}
        return observation, info

    def _next_observation(self):
        obs = self.data.iloc[
            self.current_step - self.sequence_length : self.current_step
        ][features].values
        # Convert to float32 tensor
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            logger.error(f"Invalid observation at step {self.current_step}")
            raise ValueError(f"Invalid observation at step {self.current_step}")
        return obs.cpu().numpy() #obs

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
        current_price = self.data.iloc[self.current_step]['close']

        # Execute action
        if action == 1:  # Buy
            if self.balance > 0:
                position_size = 1 #self.balance / current_price
                self.positions.append({
                    'direction': 'buy',
                    'entry_price': current_price,
                    'size': position_size
                })
                # self.balance = 0
                logger.info(f"Buy action executed at price {current_price}")
        elif action == 2:  # Sell
            if self.balance > 0:
                position_size = 1 #self.balance / current_price
                self.positions.append({
                    'direction': 'sell',
                    'entry_price': current_price,
                    'size': position_size
                })
                # self.balance = 0
                logger.info(f"Sell action executed at price {current_price}")

        # Initialize reward
        reward = 0.0

        # Update positions and calculate rewards
        positions_to_close = []
        for position in self.positions:
            position_reward, should_close = self._calculate_reward(position, current_price)
            reward += position_reward

            if should_close:
                # close position
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

        # Convert tensors to CPU for logging or NumPy conversion
        # obs_cpu = obs.cpu().numpy()

        # Additional info
        info = {}
        truncated = False
        logger.info(f"Step {self.current_step}: Reward: {reward}, Net Worth: {self.net_worth}")
        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance
        logger.info(f'Step: {self.current_step}, Balance: {self.balance:.2f}, Net Worth: {self.net_worth:.2f}, Profit: {profit:.2f}')

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
logger.info("Starting model training...")
model.learn(total_timesteps=100000)
logger.info("Model training complete")

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
logger.info("Model saved to 'ppo_forex_transformer'")

# %%
