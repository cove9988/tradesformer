import os
"""
This module provides functionalities to visualize OHLC trading data using matplotlib charts.
It contains a TradingChart class, which plots candlestick charts decorated with transaction
history lines for both winning and losing trades.
"""
"""
Class: TradingChart
-------------------
A class responsible for reading OHLC data from a CSV file, rendering candlestick charts using
mplfinance, and optionally saving the resulting plots. The class also accepts a transaction
history to plot lines representing winning and losing trades.
Attributes:
    save_plot (bool): Indicates whether the plot should be saved to a file or displayed.
    output_file (str): The file path for saving the plot image.
    ohlc (DataFrame): A DataFrame containing the time-indexed OHLC data.
    transaction_history (list): A list of dictionaries containing details about each transaction.
    parameters (dict): A dictionary of default plotting parameters passed to mplfinance.
    symbol (str): The trading symbol (e.g., currency pair or asset) extracted from the OHLC data.
"""
"""
Method: __init__(self, csv_file, transaction_history, save_plot, **kwargs)
-------------------------------------------------------------------------
Initializes the TradingChart with the given CSV file containing OHLC data, a list of
transaction details, and a flag indicating whether to save the resulting chart.
Args:
    csv_file (str): The path to the CSV file containing OHLC data.
    transaction_history (list): A list of transactions, each represented by a dictionary
        with 'pips', 'ActionTime', 'ActionPrice', 'CloseTime', 'ClosePrice', and 'CloseStep'.
    save_plot (bool): True to save the plotted chart as an image, False to display it.
    **kwargs: Additional keyword arguments that may be used for further customization.
"""
"""
Method: transaction_line(self)
-----------------------------
Generates line segments representing winning and losing trades. Accumulates and returns the
total rewards from all transactions.
Returns:
    combined_alines (list): A combined list of tuples representing the start and end points
        for all transaction lines.
    combined_colors (list): A corresponding list of colors ('b' for winning trades,
        'r' for losing trades).
    rewards (float): The total accumulated reward (sum of pips) from the transaction history.
"""
"""
Method: plot(self)
------------------
Plots the candlestick chart based on the OHLC data and overlays transaction lines for winning
and losing trades. The chart can either be saved to a specified file or displayed interactively,
depending on the save_plot attribute.
Returns:
    None
"""
import mplfinance as mpf
import pandas as pd
import datetime

class TradingChart():
    """An ohlc trading visualization using matplotlib made to render tgym environment"""
    def __init__(self, csv_file, transaction_history, save_plot, **kwargs):
        df= pd.read_csv(csv_file)
        self.save_plot =save_plot
        self.output_file = csv_file.replace("split/", "plot/").replace(".csv", ".png")
        self.ohlc = df[['time','open','high','low','close','symbol']].copy()
        self.ohlc = self.ohlc.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close'})
        self.ohlc.index = pd.DatetimeIndex(self.ohlc['Date'])
        self.transaction_history = transaction_history
        self.parameters = {"figscale":6.0,"style":"nightclouds", "type":"hollow_and_filled", "warn_too_much_data":2000 }
        self.symbol = self.ohlc.iloc[1]["symbol"]
    def transaction_line(self):
        _wlines=[]
        _llines=[]

        rewards = 0
        for tr in self.transaction_history:
            rd = tr['pips']  
            rewards += rd
            if tr['CloseStep'] >= 0 :
                if rd > 0 :
                    _wlines.append([(tr['ActionTime'],tr['ActionPrice']),(tr['CloseTime'],tr['ClosePrice'])])
                else:
                    _llines.append([(tr['ActionTime'],tr['ActionPrice']),(tr['CloseTime'],tr['ClosePrice'])])

        combined_alines = _wlines + _llines
        combined_colors = ['b'] * len(_wlines) + ['r'] * len(_llines)
        return combined_alines, combined_colors, rewards
    
    def plot(self):
        combined_alines, combined_colors, rewards = self.transaction_line()
        title = f'Symbol:{self.symbol}   Rewards:{rewards}'
        # os.makedirs(self.output_file, exist_ok=True)
        if self.save_plot:        
            dir_path = os.path.dirname(self.output_file)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            mpf.plot(
                self.ohlc, 
                type='candle', 
                alines = dict(alines=combined_alines, colors=combined_colors),
                title=title,
                savefig=dict(fname=self.output_file, dpi=300, bbox_inches="tight"),
                )
        else:    
            mpf.plot(
                self.ohlc, 
                type='candle', 
                alines = dict(alines=combined_alines, colors=combined_colors),
                title=title,
                )