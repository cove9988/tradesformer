from src.util.action_aggregation import ActionEnum
class RewardCalculator:
    def __init__(self, df, cf, shaping_reward, stop_loss, profit_taken, backward_window):
        self.df = df
        self.cf = cf
        self.shaping_reward = shaping_reward
        self.stop_loss = stop_loss
        self.profit_taken = profit_taken
        self.backward_window = backward_window
        self.reward_box = []
        self._calculate_reward()

    def _calculate_reward(self):
        _point = self.cf.symbol(self.cf.env_parameters("asset_col"), "point")
        for _step in range(self.backward_window, len(self.df)):
            buy_sl, buy_pt, sell_sl, sell_pt = False, False, False, False
            _c = self.df.iloc[_step]["close"]
            i = _step + 1
            while i < len(self.df):
                _rr = {ActionEnum.BUY: 0.0, ActionEnum.SELL: 0.0, ActionEnum.HOLD: 0.0, "step": 0}
                _h, _l = self.df.iloc[i][["high", "low"]]
                _sl_price = self.stop_loss / _point
                _pt_price = self.profit_taken / _point
                if not buy_sl and not buy_pt:
                    if _l <= _c + _sl_price:
                        buy_sl = True
                    elif _h > _c + _pt_price:
                        buy_pt = True
                    else:
                        pass
                elif buy_sl and buy_pt:
                    buy_pt = False

                if not sell_sl and not sell_pt:
                    if _h >= _c - _sl_price:
                        sell_sl = True
                    elif _l < _c - _pt_price:
                        sell_pt = True
                    else:
                        pass
                elif sell_sl and sell_pt:
                    sell_pt = False

                if (buy_pt or buy_sl) and (sell_pt or sell_sl):
                    break

                i += 1

            _sl = -self.shaping_reward * 2
            _pt = self.shaping_reward * 4
            if buy_sl:
                _rr[ActionEnum.BUY] = _sl
            elif buy_pt:
                _rr[ActionEnum.BUY] = _pt

            if sell_sl:
                _rr[ActionEnum.SELL] = _sl
            elif sell_pt:
                _rr[ActionEnum.SELL] = _pt

            if buy_sl or sell_sl:
                _rr[ActionEnum.HOLD] = -_sl
            elif buy_pt or sell_pt:
                _rr[ActionEnum.HOLD] = -_pt
            else:
                _rr[ActionEnum.HOLD] = self.shaping_reward * 0

            _rr["Step"] = _step
            self.reward_box.append(_rr)


def optimize_pt_sl(position, close_price, position_price, atr, action_signal, pt, sl):
    K, alpha, beta, gamma, delta = 1.5, 0.5, 0.8, 1.5, 0.2  # Dynamic tuning factors

    if position == "BUY":
        price_trend = close_price - position_price

        if price_trend > 0:  # Upward trend
            if action_signal == "BUY":
                # Increase PT and tighten SL
                pt = max(pt, close_price + K * atr)
                sl = max(sl, close_price - alpha * atr)
            else:  # Action signal is SELL
                # Tighten SL and reset PT
                sl = close_price - delta * atr
                pt = close_price + beta * atr
        else:  # Downward trend
            # Reduce PT and widen SL
            pt = min(pt, close_price + beta * atr)
            sl = min(sl, close_price - gamma * atr)

    elif position == "SELL":
        # Similar logic for SELL positions (inverse of BUY)
        price_trend = position_price - close_price

        if price_trend > 0:  # Downward trend
            if action_signal == "SELL":
                pt = max(pt, close_price - K * atr)
                sl = max(sl, close_price + alpha * atr)
            else:  # Action signal is BUY
                sl = close_price + delta * atr
                pt = close_price - beta * atr
        else:  # Upward trend
            pt = min(pt, close_price - beta * atr)
            sl = min(sl, close_price + gamma * atr)

    return pt, sl

pt, sl = optimize_pt_sl(
    position="BUY",
    close_price=0.7520,
    position_price=0.7500,
    atr=0.0015,
    action_signal="BUY",
    pt=0.7550,
    sl=0.7480
)
print(f"Optimized PT: {pt}, Optimized SL: {sl}")