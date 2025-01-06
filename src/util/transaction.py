from datetime import datetime
from action_enum import ActionEnum
class TransactionManager:
    def __init__(self, cf, balance, asset, stop_loss, profit_taken):
        self.cf = cf
        self.balance = balance
        self.asset = asset
        self.stop_loss = stop_loss
        self.profit_taken = profit_taken
        self.transaction_live = []
        self.transaction_history = []
        self.transaction_limit_order = []
        self.transaction_open_this_step = []
        self.transaction_close_this_step = []
        self.current_holding = 0
        self.ticket_id = 0

    def _manage_transaction(self, tr, _p, close_price, status=1):
        self.transaction_live.remove(tr)
        tr["ClosePrice"] = close_price
        tr["Point"] = int(_p)
        tr["Reward"] = int(tr["Reward"] + _p)
        tr["Status"] = status
        tr["CloseTime"] = datetime.now()
        self.balance += int(tr["Reward"])
        self.transaction_close_this_step.append(tr)
        self.transaction_history.append(tr)
        self.current_holding -= 1

    def _close_order(self, df, current_step, done):
        closed = True
        for tr in self.transaction_live:
            _point = self.cf.symbol(self.asset, "point")
            _day = df.iloc[current_step]["weekday"]
            if tr["Type"] == ActionEnum.BUY:
                _sl_price = tr["ActionPrice"] + tr["SL"] / _point
                _pt_price = tr["ActionPrice"] + tr["PT"] / _point
                if done:
                    p = (df.iloc[current_step]["Close"] - tr["ActionPrice"]) * _point
                    self._manage_transaction(tr, p, df.iloc[current_step]["Close"], status=2)
                elif df.iloc[current_step]["Low"] <= _sl_price:
                    self._manage_transaction(tr, tr["SL"], _sl_price)
                elif df.iloc[current_step]["High"] >= _pt_price:
                    self._manage_transaction(tr, tr["PT"], _pt_price)
                else:
                    closed = False

            elif tr["Type"] == ActionEnum.SELL:
                _sl_price = tr["ActionPrice"] - tr["SL"] / _point
                _pt_price = tr["ActionPrice"] - tr["PT"] / _point
                if done:
                    p = (tr["ActionPrice"] - df.iloc[current_step]["Close"]) * _point
                    self._manage_transaction(tr, p, df.iloc[current_step]["Close"], status=2)
                elif df.iloc[current_step]["High"] >= _sl_price:
                    self._manage_transaction(tr, tr["SL"], _sl_price)
                elif df.iloc[current_step]["Low"] <= _pt_price:
                    self._manage_transaction(tr, tr["PT"], _pt_price)
                else:
                    closed = False
        return closed

