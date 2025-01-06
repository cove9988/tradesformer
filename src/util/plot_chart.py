import mplfinance as mpf
import pandas as pd
import datetime

class TradingChart( ):
    """An ohlc trading visualization using matplotlib made to render tgym environment"""
    def __init__(self, weekly_data_week:int, transaction_file, **kwargs):
        df = pd.read_csv(f'./data/weekly_data_week_{weekly_data_week}.csv')
        transaction_history = pd.read_csv('./data/log/transaction_file.csv').apply(lambda x: x.str.strip() if x.dtype == "object" else x).to_dict('records')
        self.ohlc = df[['Datetime','Open','High','Low','Close']].copy()
        self.ohlc = self.ohlc.rename(columns={'Datetime':'Date'})
        self.ohlc.index = pd.DatetimeIndex(self.ohlc['Date'])
        self.transaction_history = transaction_history
        self.parameters = {"figscale":6.0,"style":"nightclouds", "type":"hollow_and_filled", "warn_too_much_data":2000 }
        self.symbols =['EURUSD'] #self.ohlc['symbol'].unique()
    def transaction_line(self, symbol):
        _wlines=[]
        _wcolors=[]
        _llines=[]
        _lcolors=[]

        rewards = 0
        b_count, s_count = 0, 0
        for tr in self.transaction_history:
            if tr["currency_pair"] == symbol : 
                rd = tr['pips']  
                rewards += rd
                if tr['close_price'] > 0 :
                    if tr['direction'] == 'b' :
                        b_count += 1
                        if rd > 0 :
                            _wlines.append([(tr['open_time'],tr['open_price']),(tr['close_time'],tr['close_price'])])
                            _wcolors.append('c')
                        else:
                            _llines.append([(tr['open_time'],tr['open_price']),(tr['close_time'],tr['close_price'])])
                            _lcolors.append('c')
                    elif tr['Type'] == 's' :
                        s_count += 1
                        if rd > 0 :
                            _wlines.append([(tr['open_time'],tr['open_price']),(tr['close_time'],tr['close_price'])])
                            _wcolors.append('k')
                        else:
                            _llines.append([(tr['open_time'],tr['open_price']),(tr['close_time'],tr['close_price'])])
                            _lcolors.append('k')
        return _wlines, _wcolors,_llines, _lcolors, rewards, b_count, s_count
    
    def plot(self):
        for s in self.symbols:
            _wlines, _wcolors,_llines, _lcolors, rewards, b_count, s_count = self.transaction_line(s)
            _wseq = dict(alines=_wlines, colors=_wcolors)
            _lseq = dict(alines=_llines, colors=_lcolors, linestyle='--')
            _ohlc = self.ohlc.query(f'symbol=="{s}"')
            _style = mpf.make_mpf_style(base_mpl_style='seaborn',rc={'axes.grid':True})
            fig = mpf.figure(style=_style,figsize=(40,20))
            ax1 = fig.subplot()
            ax2 = ax1.twinx()
            title = f'{s} reward:{rewards} Buy:{b_count} Sell:{s_count}'
            mpf.plot(_ohlc, alines=_lseq ,mav=(10,20), ax=ax1,type='ohlc',style='default')
            mpf.plot(_ohlc,alines=_wseq, ax=ax2,type='candle',style='yahoo',axtitle=title)
            fig.savefig(f'./data/log/{s}-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}')
  