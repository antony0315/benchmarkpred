import pandas as pd
import numpy as np
#import plotly 
#import plotly.io as pio
# import plotly.graph_objects as go
# from plotly.offline import init_notebook_mode, iplot
#pio.renderers.default='browser' #'svg' or 'browser'
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from datetime import datetime
import talib
# import seaborn as sns
# %matplotlib inline

df=pd.read_csv('C:/Users/anton/OneDrive/桌面/newpreddata.csv')

df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['RSI']=talib.RSI(df['close'], timeperiod=14)
df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
df['fastk'], df['fastd'] = talib.STOCHF(df['high'], df['low'], df['close'], fastk_period=5, fastd_period=3, fastd_matype=0)
df['sma5']=talib.SMA(df['close'],timeperiod=5)
df['sma20']=talib.SMA(df['close'],timeperiod=20)
df['sma60']=talib.SMA(df['close'],timeperiod=60)
df['sma5-sma60']=df['sma5']-df['sma60']
df['max']=talib.MAX(df['high'],timeperiod=20)
df['close-max']=np.sign(df['close']-df['max'])*np.log(np.abs(df['close']-df['max']))
df['willr']=talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
df['ultosc']=talib.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
df['rocr'] = talib.ROCR100(df['close'], timeperiod=10)
df['apo']=talib.APO(df['close'])
df['sar']=df['close']-talib.SAR(df['high'],df['low'])
df['vwap5']=(df['close']*df['volumn']).rolling(5).sum()/(df['volumn']).rolling(5).sum()
df['vwap20']=(df['close']*df['volumn']).rolling(20).sum()/(df['volumn']).rolling(20).sum()
df['vwap5-20']=df['vwap5']-df['vwap20']
df['close-ma']=(df['close']-df['sma60'])/df['sma60']
df['close-ma20']=df['close']-df['sma20']
df['bull-bear']=df['bull']-df['bear']
#ForceIndex
p01=df['forceindex'].quantile(0.05)
p99=df['forceindex'].quantile(0.95)
df['forceindex']=df['forceindex'].clip(p01,p99)
#
df['sma5slop']=df['sma5'].diff(1)
df['sma10']=talib.SMA(df['close'],timeperiod=10)
df['close-sma10']=df['close']-df['sma10']



df.to_csv('C:/Users/anton/OneDrive/桌面/newpreddata2.csv')












