# %%
import logging
from src.ppo_model import single_csv_training, multiply_csv_files_traning
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

csv_file = "/home/paulg/github/tradesformer/data/split/AUDUSD/weekly/AUDUSD_2022_1.csv"
single_csv_training(csv_file=csv_file)
# data_directory = "/home/paulg/github/tradesformer/data/split/EURUSD/weekly"
# multiply_csv_files_traning(data_directory=data_directory)

# %%
