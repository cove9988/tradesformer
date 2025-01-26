from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TrainingMetricsCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.sharpe_ratios = []
        self.drawdowns = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Track metrics only when episodes complete
        if "sharpe" in self.locals['infos'][0] and "max_drawdown" in self.locals['infos'][0]:
            self.episode_count += 1
            self.sharpe_ratios.append(self.locals['infos'][0]['sharpe'])
            self.drawdowns.append(self.locals['infos'][0]['max_drawdown'])
            
            # Log to tensorboard every N episodes
            if self.episode_count % 10 == 0:
                self.logger.record('train/mean_sharpe', np.mean(self.sharpe_ratios[-10:]))
                self.logger.record('train/max_drawdown', np.mean(self.drawdowns[-10:]))
                self.logger.record('train/episodes', self.episode_count)
        
        return True