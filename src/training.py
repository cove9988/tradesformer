import os
import datetime
import logging
import glob
import torch.nn as nn
from stable_baselines3 import PPO
from src.ppo_model import ForexTradingEnv, CustomCombinedExtractor
from src.util.read_config import EnvConfig
from src.util.logger_config import setup_logging
logger = logging.getLogger(__name__)

def multiply_csv_files_traning(data_directory,env_config_file,asset, sequence_length=24):
    # Define the directory containing the CSV files
    cf = EnvConfig(env_config_file)
    features = cf.env_parameters("observation_list")
    print(features)
    
    # Get a list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(data_directory, "*.csv"))
    # Set up PPO model parameters
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(sequence_length=sequence_length),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        activation_fn=nn.ReLU
    )

    # Initialize the batch counter
    batch_num = 1
    model = None
    # Loop through each CSV file for training
    for file in csv_files:
        # Preprocess the data (scaling, etc.)
        # data[features] = scaler.fit_transform(data[features])

        # Reset the environment for the new file
        env = ForexTradingEnv(file, cf, asset, features, sequence_length)
        # lr_schedule = LearningRateSchedule(linear_schedule(1e-4))
        lr_schedule = 1e-4
        # Train the model on the current file
        logger.info(f"Starting training on file {file} (Batch {batch_num})")
        model_filename = file.replace("split/", "model/").replace(".csv", ".zip")

        # model_filename = f'/home/paulg/github/tradesformer/data/model/ppo_forex_transformer_batch_{batch_num}.zip'
        if not model :
            # Initial model training
            model = PPO(
                'MlpPolicy',
                env,
                verbose=1,
                policy_kwargs=policy_kwargs,
                learning_rate=lr_schedule,  # Reduced learning rate
                max_grad_norm=0.5    # Gradient clipping
            )
        model.learn(total_timesteps=100000)  # Adjust the number of timesteps per batch as needed

        # Save the model after training on this file
        model.save(model_filename)
        logger.info(f"Model saved as {model_filename}")

        # Increment the batch number for the next file
        batch_num += 1

        # Reload the model if needed, for the next batch (optional, if you want to continue learning)
        model = PPO.load(model_filename, env=env, learning_rate=lr_schedule )

    logger.info("Finished training on all files")

if __name__ == "__main__":
    asset = "AUDUSD"      
    data_directory = "/home/paulg/github/tradesformer/data/split/AUDUSD/weekly"
    env_config_file ='/home/paulg/github/tradesformer/src/configure.json'
    sequence_length=24
    setup_logging(asset=asset, console_level=logging.ERROR, file_level=logging.INFO)
   
    multiply_csv_files_traning(data_directory=data_directory, 
                               env_config_file=env_config_file, 
                               asset=asset, 
                               sequence_length=sequence_length)

# %%
