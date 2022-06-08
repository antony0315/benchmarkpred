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

pred=pd.read_csv('C:/Users/anton/OneDrive/桌面/pred.csv',names=['pred','benchmark'])

fig = go.Figure()
fig.add_scattergl(x=pred.index,y=pred['benchmark'], line={'color': 'black'})
fig.add_scattergl(x=pred.index, y=pred.benchmark.where(pred.pred == 1), line={'color': 'red'})
iplot(fig)