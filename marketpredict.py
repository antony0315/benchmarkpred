import pandas as pd
import numpy as np
import plotly 
import plotly.io as pio
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
pio.renderers.default='browser' #'svg' or 'browser'
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import talib
import seaborn as sns
%matplotlib inline
df=pd.read_csv('C:/Users/user/Desktop/bench.csv')
df.columns=['時間', '開盤價', '最高價', '最低價', '收盤價', '成交量', 'target']
plot=pd.read_csv('C:/Users/user/Desktop/bullbear.csv')
df['時間'] = pd.to_datetime(df['時間'], format="%Y/%m/%d")
plot['Date']=pd.to_datetime(plot['Date'],format="%Y-%m-%d")
forceindex=pd.read_csv('C:/Users/user/Desktop/force.csv')
forceindex['時間']=pd.to_datetime(forceindex['時間'],format="%Y/%m/%d")
ntddollar=pd.read_csv('C:/Users/user/Desktop/ntddollar.csv')
ntddollar['時間']=pd.to_datetime(ntddollar['時間'],format="%Y/%m/%d")
#視覺化
fig = go.Figure()
fig.add_scattergl(x=df['時間'], y=df['收盤價'], line={'color': 'black'})
fig.add_scattergl(x=df['時間'], y=df.收盤價.where(df.target == 1), line={'color': 'red'})
iplot(fig)

#增加特徵
df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['收盤價'], fastperiod=12, slowperiod=26, signalperiod=9)
df['RSI']=talib.RSI(df['收盤價'], timeperiod=14)
df['slowk'], df['slowd'] = talib.STOCH(df['最高價'], df['最低價'], df['收盤價'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
df['fastk'], df['fastd'] = talib.STOCHF(df['最高價'], df['最低價'], df['收盤價'], fastk_period=5, fastd_period=3, fastd_matype=0)
df['sma5']=talib.SMA(df['收盤價'],timeperiod=5)
df['sma20']=talib.SMA(df['收盤價'],timeperiod=20)
df['sma60']=talib.SMA(df['收盤價'],timeperiod=60)
df['sma5-sma60']=df['sma5']-df['sma60']
df['max']=talib.MAX(df['最高價'],timeperiod=20)
df['close-max']=np.sign(df['收盤價']-df['max'])*np.log(np.abs(df['收盤價']-df['max']))
df['willr']=talib.WILLR(df['最高價'], df['最低價'], df['收盤價'], timeperiod=14)
df['ultosc']=talib.ULTOSC(df['最高價'], df['最低價'], df['收盤價'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
df['rocr'] = talib.ROCR100(df['收盤價'], timeperiod=10)
df['apo']=talib.APO(df['收盤價'])
df['sar']=df['收盤價']-talib.SAR(df['最高價'],df['最低價'])
df['vwap5']=(df['收盤價']*df['成交量']).rolling(5).sum()/(df['成交量']).rolling(5).sum()
df['vwap20']=(df['收盤價']*df['成交量']).rolling(20).sum()/(df['成交量']).rolling(20).sum()
df['vwap5-20']=df['vwap5']-df['vwap20']
df['close-ma']=(df['收盤價']-df['sma60'])/df['sma60']
df['close-ma20']=df['收盤價']-df['sma20']
df=pd.merge(df,plot,left_on='時間',right_on='Date',how='left').drop('Date',axis=1)
df['bull-bear']=df['bull']-df['bear']
#ForceIndex
df=pd.merge(df,forceindex,on='時間',how='left')
p01=df['forceindex'].quantile(0.05)
p99=df['forceindex'].quantile(0.95)
df['ForceIndex']=df['forceindex'].clip(p01,p99)
#
df['sma5slop']=df['sma5'].diff(1)
df=pd.merge(df,ntddollar,on='時間',how='left')


df.to_csv('0504.csv')

#%%
for i in df.keys():
    fig,ax = plt.subplots()
    ax.hist(df.loc[df['target']>0,[i]],bins=50,color='red',alpha=0.7)
    ax2=ax.twinx()
    ax2.hist(df.loc[df['target']<0,[i]],bins=50,color='green',alpha=0.7)
    plt.title(i)


df.to_csv('0503df.csv')

#%%
i='sma5slop'
fig,ax = plt.subplots()
ax.hist(df.loc[df['target']>0,[i]],bins=100,color='red',alpha=0.7)
ax2=ax.twinx()
ax2.hist(df.loc[df['target']<0,[i]],bins=100,color='green',alpha=0.7)
plt.title(i)
print(df.loc[df['target']>0,[i]].mean())
print(df.loc[df['target']<0,[i]].mean())




#%%
dfcorr=df.corr()
fig,ax=plt.subplots(figsize=(9,9))
sns.heatmap(dfcorr,annot=True,vmax=1,vmin=0,xticklabels=True,yticklabels=True,square=True,cmap="Blues")

























