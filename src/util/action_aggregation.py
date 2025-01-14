from collections import deque
import numpy as np

class ActionEnum(enumerate):
    """
    Enum for action space
    :param enumerate: Enumerate class
    """
    Hold = 0
    Buy = 1
    Sell = 2
    Exit = 3

class ActionAggregator:
    
    def __init__(self, window_size =6): #1hr
        self.window_size = window_size
        self.action_window = deque(maxlen=window_size)
        self.stability_tracker = ActionStabilityTracker()
        self.reward = 0
    def add_action(self, action):
        """Add the latest action to the window."""
        self.action_window.append(action)
        self.reward = self.stability_tracker.calculate_stability_reward(action)
        
        """Decide action based on the aggregated signals in the window."""
        if len(self.action_window) < self.window_size:
            return ActionEnum.Hold, self.reward  # Not enough data yet

        # Count actions
        buy_count = self.action_window.count(ActionEnum.Buy)
        sell_count = self.action_window.count(ActionEnum.Sell)
        hold_count = self.action_window.count(ActionEnum.Hold)

        # Decide based on majority
        total = len(self.action_window)
        if buy_count / total > 0.7:  # 70% threshold for Buy
            return ActionEnum.Buy, self.reward
        elif sell_count / total > 0.7:  # 70% threshold for Sell
            return ActionEnum.Sell, self.reward
        else:
            return ActionEnum.Hold, self.reward
        
class ActionAggregatorOptimized:
    def __init__(self, base_window_size=12, volatility_threshold=0.01):
        self.base_window_size = base_window_size
        self.volatility_threshold = volatility_threshold
        self.action_window = deque(maxlen=base_window_size)
        self.weighted_window = deque(maxlen=base_window_size)

    def adjust_window_size(self, volatility):
        """Adjust the window size dynamically based on volatility."""
        if volatility > self.volatility_threshold:
            self.action_window = deque(maxlen=self.base_window_size // 2)
            self.weighted_window = deque(maxlen=self.base_window_size // 2)
        else:
            self.action_window = deque(maxlen=self.base_window_size)
            self.weighted_window = deque(maxlen=self.base_window_size)

    def add_action(self, action, weight=1.0):
        """Add the latest action and its weight to the window."""
        self.action_window.append(action)
        self.weighted_window.append(weight)

    def calculate_price_trend(self, prices):
        """Calculate a simple trend indicator (e.g., moving average difference)."""
        if len(prices) < 5:  # Ensure enough data
            return 0
        short_ma = np.mean(prices[-3:])
        long_ma = np.mean(prices[-5:])
        return short_ma - long_ma

    def get_aggregated_action(self, prices):
        """Decide the action based on aggregated signals and price trends."""
        if len(self.action_window) < self.action_window.maxlen:
            return ActionEnum.Hold  # Not enough data yet

        # Count weighted actions
        buy_weight = sum(w for a, w in zip(self.action_window, self.weighted_window) if a == ActionEnum.Buy)
        sell_weight = sum(w for a, w in zip(self.action_window, self.weighted_window) if a == ActionEnum.Sell)
        hold_weight = sum(w for a, w in zip(self.action_window, self.weighted_window) if a == ActionEnum.Hold)

        # Price trend adjustment
        trend = self.calculate_price_trend(prices)

        # Decide based on weighted majority and trend
        if buy_weight > sell_weight and trend > 0:
            return ActionEnum.Buy
        elif sell_weight > buy_weight and trend < 0:
            return ActionEnum.Sell
        else:
            return ActionEnum.Hold

class ActionStabilityTracker:
    def __init__(self, consistency_reward=0.01):
        self.last_action = None
        self.consecutive_count = 0
        self.change_penalty = -consistency_reward * 2   # Penalty for frequent changes
        self.consistency_reward = consistency_reward  # Reward for consistent actions
        self.max_consistency = 12 # if consistency no limit, RL will take adv of one directly without any changes to accumulate rewards.

    def calculate_stability_reward(self, current_action):
        """Calculate reward/penalty based on action stability.
        if continue 
        """
        if current_action == self.last_action:
            self.consecutive_count += 1
            if self.consecutive_count < self.max_consistency:
                reward = self.consistency_reward * self.consecutive_count
            else:
                reward = self.consistency_reward * self.change_penalty
        else:
            reward = self.change_penalty
            self.consecutive_count = 1  # Reset the counter for new action

        self.last_action = current_action
        return reward


# Integrate into PPO reward
def calculate_ppo_reward(current_action, stability_tracker):
    """Calculate final reward incorporating base reward and stability."""
    stability_reward = stability_tracker.calculate_stability_reward(current_action)
    return stability_reward


# Example usage
if __name__ == "__main__":
    # Simulate price data and actions
    actions = [0, 1, 2, 0, 1, 2, 0, 1, 1, 1,1, 1, 1, 1, 1,2, 1, 1, 0]
    volatilities = [0.009, 0.012, 0.008, 0.015, 0.007]
    prices = [1.1010, 1.1020, 1.1030, 1.1015, 1.1040, 1.1050] 
    aggregator = ActionAggregator()

    for i, action in enumerate(actions):
        # Get aggregated action
        aggregated_action, reward = aggregator.add_action(action)
        print(f"simple Step {i + 1}: Aggregated Action: {aggregated_action} {reward}")
        
    aggregator = ActionAggregatorOptimized(base_window_size=10, volatility_threshold=0.01)

    for i, action in enumerate(actions):
        # Simulate volatility adjustment
        volatility = volatilities[i % len(volatilities)]
        aggregator.adjust_window_size(volatility)

        # Add action with weight
        weight = 1.5 if i > len(actions) // 2 else 1.0  # Example weight strategy
        aggregator.add_action(action, weight)

        # Get aggregated action
        aggregated_action = aggregator.get_aggregated_action(prices[: i + 1])
        print(f"Step {i + 1}: Aggregated Action: {aggregated_action}")


    base_rewards = [10, 15, 5, 20, 25, -5, 30, 35, 10]  # Example base rewards

    stability_tracker = ActionStabilityTracker()
    total_rewards = []

    for action, base_reward in zip(actions, base_rewards):
        final_reward = calculate_ppo_reward(action, stability_tracker)
        total_rewards.append(final_reward)
        print(f"Action: {action}, Base Reward: , Final Reward: {final_reward}")

    print("Total Rewards:", total_rewards)