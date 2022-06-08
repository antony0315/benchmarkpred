import pandas_datareader.data as web
import pandas as pd
import datetime as dt
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import style
import talib
# %matplotlib inline
# import plotly 
# import plotly.io as pio
# import plotly.graph_objects as go
# from plotly.offline import init_notebook_mode, iplot
#pio.renderers.default='browser' #'svg' or 'browser'


tw100=pd.read_csv('C:/Users/anton/OneDrive/桌面/TW100.csv',header=None)

start=dt.datetime(2022,1,10)
end=dt.date.today()

df=pd.DataFrame()
for i in tw100[0]:
    stock=str(i)+'.TW'
    price=web.DataReader(stock,"yahoo",start,end)['Adj Close']
    price=pd.DataFrame({stock:price})
    df=pd.concat([df,price],axis='columns')
df=df.fillna(method='ffill')



sma=pd.DataFrame()
for i in df.keys():
    sma[i+'sma5']=talib.SMA(df[i],timeperiod=5)
    sma[i+'sma10']=talib.SMA(df[i],timeperiod=10)
    sma[i+'sma20']=talib.SMA(df[i],timeperiod=20)

bull=pd.DataFrame()
bear=pd.DataFrame()
for j in range(100):
        a=3*j
        b=a+1
        c=b+1
        bull[j]=np.where((sma.iloc[:,a] > sma.iloc[:,b]) & (sma.iloc[:,b] > sma.iloc[:,c])
                             , 1,0)
        bear[j]=np.where((sma.iloc[:,a] < sma.iloc[:,b]) & (sma.iloc[:,b] <sma.iloc[:,c])
                             , 1,0)

plot=pd.DataFrame()
plot['bull']=bull.apply(lambda x:x.sum(),axis=1)
plot['bear']=bear.apply(lambda x:x.sum(),axis=1)
plot.index=df.index.copy()

# line=go.Scatter(x=plot.index,y=plot['bull'],name='bull')
# line2=go.Scatter(x=plot.index,y=plot['bear'],name='bear')
# fig=go.Figure([line,line2])
# fig.show()
#plot.to_csv('bullbear.csv')



