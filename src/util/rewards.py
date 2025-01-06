from action_enum import ActionEnum
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
            _c = self.df.iloc[_step]["Close"]
            i = _step + 1
            while i < len(self.df):
                _rr = {ActionEnum.BUY: 0.0, ActionEnum.SELL: 0.0, ActionEnum.HOLD: 0.0, "Step": 0}
                _h, _l = self.df.iloc[i][["High", "Low"]]
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
