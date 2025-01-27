import logging
import datetime
import torch.nn as nn
from stable_baselines3 import PPO
from src.trading_gym_env import ForexTradingEnv, CustomCombinedExtractor
from src.util.read_config import EnvConfig
from src.util.logger_config import setup_logging
from src.util.training_callback import TrainingMetricsCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv

logger = logging.getLogger(__name__)

def linear_schedule(start_lr = 3e-4, end_lr = 1e-5, total_timesteps=1e6):
    lr_schedule = get_schedule_fn(1e-4)  # For constant LR
    # OR for decaying LR:
    lr_schedule = lambda progress: start_lr - (start_lr - end_lr) * progress
    return lr_schedule

def single_csv_training(csv_file, env_config_file, asset, model_name ='', cf = None, number_envs = 1):
    features = cf.env_parameters("observation_list")
    sequence_length = cf.env_parameters("backward_window")
    print(features)
    lr_schedule = linear_schedule(3e-4, 1e-5, total_timesteps=1e6)
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(sequence_length=sequence_length),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.ReLU,
        ortho_init=False # better for finacial data
    )
    # env = ForexTradingEnv(csv_file, cf, asset, features=features, sequence_length=sequence_length, logger_show= True)
    env = DummyVecEnv([lambda: ForexTradingEnv(csv_file, cf, asset, features=features, sequence_length=sequence_length, logger_show= True) for _ in range(number_envs)])
    # env.logger_show = True
    if model_name:
        model = PPO.load(model_name, env=env, learning_rate=lr_schedule)
    else:
        model = PPO(
            # 'CnnPolicy' , # support GPU
            'MlpPolicy', # CPU only
            env,
            device='cuda',
            verbose=1,
            vf_coef=0.5,  # Increase value loss weight
            target_kl=0.05,  # Add target KL for early stopping
            normalize_advantage=True,            
            policy_kwargs=policy_kwargs,
            learning_rate=lr_schedule,  # Reduced learning rate
            max_grad_norm=0.5    # Gradient clipping
        )

    # Train the agent
    logger.info("Starting model training...")
    callback = TrainingMetricsCallback()
    model.learn(
        total_timesteps=500000,
        callback=callback,
        tb_log_name=f"./data/log/{asset}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    logger.info("Model training complete")
    model_filename = csv_file.replace("split/", "model/").replace(".csv", "_single_test.zip")
    model.save(model_filename)
if __name__ == "__main__":
    asset = "EURUSD"
    env_config_file = './src/configure.json'
    cf = EnvConfig(env_config_file)
    split_cfg = cf.data_processing_parameters("train_eval_split")
    base_path = split_cfg["base_path"].format(symbol=asset)
    csv_file = f"{base_path}/{split_cfg["train_dir"]}/{asset}_2022_22.csv"
    model_name = '' #f'./data/model/{asset}/weekly/{asset}_2023_71'
    setup_logging(asset =asset, console_level=logging.CRITICAL, file_level=logging.INFO)
    single_csv_training(csv_file=csv_file, env_config_file =env_config_file, asset= asset, model_name=model_name, cf=cf, number_envs=4)
    
    

