import mplfinance as mpf
import pandas as pd
import datetime

class TradingChart():
    """An ohlc trading visualization using matplotlib made to render tgym environment"""
    def __init__(self, df, transaction_history, **kwargs):
        self.ohlc = df[['time','open','high','low','close','symbol']].copy()
        self.ohlc = self.ohlc.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close'})
        self.ohlc.index = pd.DatetimeIndex(self.ohlc['Date'])
        self.transaction_history = transaction_history
        self.parameters = {"figscale":6.0,"style":"nightclouds", "type":"hollow_and_filled", "warn_too_much_data":2000 }
        self.symbol = self.transaction_history[0]["Symbol"]
    def transaction_line(self):
        _wlines=[]
        _llines=[]

        rewards = 0
        for tr in self.transaction_history:
            rd = tr['Reward']  
            rewards += rd
            if tr['CloseStep'] >= 0 :
                if tr['Type'] == 'Buy' :
                    if rd > 0 :
                        _wlines.append([(tr['ActionTime'],tr['ActionPrice']),(tr['CloseTime'],tr['ClosePrice'])])
                    else:
                        _llines.append([(tr['ActionTime'],tr['ActionPrice']),(tr['CloseTime'],tr['ClosePrice'])])
                elif tr['Type'] == 'Sell' :
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
        output_file = f'./data/plot/{self.symbol}-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.png'
        mpf.plot(
            self.ohlc, 
            type='candle', 
            alines = dict(alines=combined_alines, colors=combined_colors),
            title=title,
            savefig=dict(fname=output_file, dpi=300, bbox_inches="tight"),
            )
            
# import mplfinance as mpf
# import pandas as pd
# import datetime
# import logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# class TradingChart( ):
#     """An ohlc trading visualization using matplotlib made to render tgym environment"""
#     def __init__(self, df, transaction_history, **kwargs):
#         # transaction_history = pd.read_csv('./data/log/transaction_file.csv').apply(lambda x: x.str.strip() if x.dtype == "object" else x).to_dict('records')
#         self.ohlc = df[['time','open','high','low','close','symbol']].copy()
#         self.ohlc = self.ohlc.rename(columns={'time':'Date', 'open':'Open', 'high':'High','low':'Low','close':'Close'})
#         self.ohlc.index = pd.DatetimeIndex(self.ohlc['Date'])
#         self.transaction_history = transaction_history
#         self.parameters = {"figscale":6.0,"style":"nightclouds", "type":"hollow_and_filled", "warn_too_much_data":2000 }
#         self.symbol =transaction_history[0]["Symbol"]
#     def transaction_line(self, symbol):
#         _wlines=[]
#         _wcolors=[]
#         _llines=[]
#         _lcolors=[]

#         rewards = 0
#         b_count, s_count = 0, 0
#         for tr in self.transaction_history:
#             if tr["Symbol"] == symbol : 
#                 rd = tr['pips']  
#                 rewards += rd
#                 if tr['ClosePrice'] > 0 :
#                     if tr['Type'] == 'b' :
#                         b_count += 1
#                         if rd > 0 :
#                             _wlines.append([(tr['ActionTime'],tr['ActionPrice']),(tr['CloseTime'],tr['ClosePrice'])])
#                             _wcolors.append('c')
#                         else:
#                             _llines.append([(tr['ActionTime'],tr['ActionPrice']),(tr['CloseTime'],tr['ClosePrice'])])
#                             _lcolors.append('c')
#                     elif tr['Type'] == 's' :
#                         s_count += 1
#                         if rd > 0 :
#                             _wlines.append([(tr['ActionTime'],tr['ActionPrice']),(tr['CloseTime'],tr['ClosePrice'])])
#                             _wcolors.append('k')
#                         else:
#                             _llines.append([(tr['ActionTime'],tr['ActionPrice']),(tr['CloseTime'],tr['ClosePrice'])])
#                             _lcolors.append('k')
#         return _wlines, _wcolors,_llines, _lcolors, rewards, b_count, s_count
    
#     def plot(self):
#         _wlines, _wcolors,_llines, _lcolors, rewards, b_count, s_count = self.transaction_line(self.symbol)
#         _wseq = dict(alines=_wlines, colors=_wcolors)
#         _lseq = dict(alines=_llines, colors=_lcolors, linestyle='--')
#         _ohlc = self.ohlc.query(f'symbol=="{self.symbol}"')
#         _style = mpf.make_mpf_style(base_mpl_style='base_mpl_style',rc={'axes.grid':True})
#         fig = mpf.figure(style=_style,figsize=(40,20))
#         ax1 = fig.subplot()
#         ax2 = ax1.twinx()
#         title = f'{self.symbol} reward:{rewards} Buy:{b_count} Sell:{s_count}'
#         mpf.plot(_ohlc, alines=_lseq ,mav=(10,20), ax=ax1,type='ohlc',style='default')
#         mpf.plot(_ohlc,alines=_wseq, ax=ax2,type='candle',style='yahoo',axtitle=title)
#         fig.savefig(f'./data/log/{self.symbol}-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}')
  