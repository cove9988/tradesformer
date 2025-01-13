import logging
import datetime
import torch.nn as nn
from stable_baselines3 import PPO
from src.ppo_model import ForexTradingEnv, CustomCombinedExtractor
from src.util.read_config import EnvConfig
from src.util.logger_config import setup_logging
logger = logging.getLogger(__name__)

def single_csv_training(csv_file, env_config_file, asset):
    cf = EnvConfig(env_config_file)
    features = cf.env_parameters("observation_list")
    sequence_length = cf.env_parameters("backward_window")
    print(features)
    # lr_schedule = LearningRateSchedule(linear_schedule(1e-4))  # Start with 1e-4
    lr_schedule = 1e-4
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(sequence_length=sequence_length),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        activation_fn=nn.ReLU
    )
    env = ForexTradingEnv(csv_file, cf, asset, features=features, sequence_length=sequence_length, logger_show= True)
    env.logger_show = True
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,  # Reduced learning rate
        max_grad_norm=0.5    # Gradient clipping
    )

    # Train the agent
    logger.info("Starting model training...")
    model.learn(total_timesteps=10000)
    logger.info("Model training complete")


if __name__ == "__main__":
<<<<<<< HEAD
    asset = "EURUSD"    
    csv_file = f"./data/split/{asset}/weekly/{asset}_2023_1.csv"
    env_config_file ='./src/configure.json'
=======
    asset = "AUDUSD"    
    csv_file = f"/home/paulg/github/tradesformer/data/split/{asset}/weekly/{asset}_2022_1.csv"
    env_config_file ='/home/paulg/github/tradesformer/src/configure.json'
>>>>>>> upstream/mac_v
    setup_logging(asset =asset, console_level=logging.WARNING, file_level=logging.INFO)
    single_csv_training(csv_file=csv_file, env_config_file =env_config_file, asset= asset)

