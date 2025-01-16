from collections import deque
import numpy as np

class ActionEnum(enumerate):
    """
    ActionEnum is an enumeration for different types of actions in a trading environment.

    Attributes:
        Hold (int): Represents holding the current position, value is 0.
        Buy (int): Represents buying an asset, value is 1.
        Sell (int): Represents selling an asset, value is 2.
        Exit (int): Represents exiting the trading position, value is 3.
    """
    Hold = 0
    Buy = 1
    Sell = 2
    Exit = 3

class ActionAggregator:
    """
    A class used to aggregate actions over a specified window size and decide on a final action based on the majority.
    Attributes
    ----------
    window_size : int
        The size of the window to aggregate actions over (default is 6).
    action_window : deque
        A deque to store the actions within the window.
    stability_tracker : ActionStabilityTracker
        An instance of ActionStabilityTracker to calculate stability rewards.
    reward : float
        The reward calculated based on the stability of the actions.
    Methods
    -------
    add_action(action)
        Adds the latest action to the window and decides the final action based on the aggregated signals.
    """
    
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
    """
    A class to aggregate actions based on a dynamic window size and price trends.
    Attributes:
        base_window_size (int): The base size of the action window.
        volatility_threshold (float): The threshold of volatility to adjust the window size.
        action_window (deque): A deque to store recent actions.
        weighted_window (deque): A deque to store weights of recent actions.
    Methods:
        __init__(base_window_size=12, volatility_threshold=0.01):
            Initializes the ActionAggregatorOptimized with the given base window size and volatility threshold.
        adjust_window_size(volatility):
            Adjusts the window size dynamically based on the given volatility.
        add_action(action, weight=1.0):
            Adds the latest action and its weight to the window.
        calculate_price_trend(prices):
            Calculates a simple trend indicator (e.g., moving average difference) based on the given prices.
        get_aggregated_action(prices):
            Decides the action based on aggregated signals and price trends.
    """
    
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
    """
    A class to track the stability of actions and calculate rewards or penalties based on the consistency of actions.
    Attributes:
        last_action (any): The last action taken.
        consecutive_count (int): The count of consecutive identical actions.
        change_penalty (float): The penalty applied for changing actions frequently.
        consistency_reward (float): The reward given for consistent actions.
        max_consistency (int): The maximum count of consecutive actions to limit the reward.
    Methods:
        __init__(consistency_reward=0.01):
            Initializes the ActionStabilityTracker with a specified consistency reward.
        calculate_stability_reward(current_action):
            Calculates the reward or penalty based on the stability of the current action.
            Args:
                current_action (any): The current action to evaluate.
            Returns:
                float: The calculated reward or penalty based on action stability.
    """
    
    def __init__(self, consistency_reward=0.01):
        self.last_action = None
        self.consecutive_count = 0
        self.change_penalty = -consistency_reward * 2   # Penalty for frequent changes
        self.consistency_reward = consistency_reward  # Reward for consistent actions
        self.max_consistency = 12 # if consistency no limit, RL will take adv of one directly without any changes to accumulate rewards.


    def calculate_stability_reward(self, current_action ):
        """
        Calculate the reward or penalty based on the stability of the current action.

        Args:
            current_action (any): The current action to evaluate.

        Returns:
            float: The calculated reward or penalty based on action stability.
        """
        if current_action == self.last_action:
            self.consecutive_count += 1
            if self.consecutive_count < self.max_consistency:
                reward = self.consistency_reward * self.consecutive_count
            else:
                reward = self.change_penalty * self.consecutive_count
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