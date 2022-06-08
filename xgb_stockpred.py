
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%資料前處裡
df=pd.read_csv('C:/Users/anton/OneDrive/桌面/0512.csv')
x_train,x_test,y_train,y_test=train_test_split(df,df.target,test_size=0.2)
x_train=x_train.drop('target',axis=1)
x_test=x_test.drop('target',axis=1)

y_train=y_train.replace(-1,0)
y_test=y_test.replace(-1,0)
import xgboost as xgb
from sklearn.metrics import log_loss
dtrain=xgb.DMatrix(x_train.values,y_train.values)
dvalid=xgb.DMatrix(x_test.values,y_test.values)
cvdata=xgb.DMatrix(df.drop('target',axis=1),df.target.replace(-1,0))
#%%建模 
params={'objective':'binary:logistic','silent':1,'random_state':71}
        #目標函數: #回歸任務:'reg:squarederror'    最小化MSE
                  #二元分類:'binary:logistic'     最小化logloss
                  #多元分類:'multi:softprob'      最小化multi-class logloss
num_round=50
#在watchlist中組合訓練與驗證資料
watchlist=[(dtrain,'train'),(dvalid,'eval')]
#訓練模型，並監控分數變化
model=xgb.train(params,dtrain,num_round,evals=watchlist)
#計算valid的logloss分數
va_pred=model.predict(dvalid)
score=log_loss(y_test,va_pred)
print(f'logloss:{score:.4f}')
#對test_y預測   輸出值為機率

#%%評價
y_pred=[]
for i in range(len(va_pred)):
    if va_pred[i]>0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
y_test=np.array(y_test)
y_accuracy=[]
for i in range(len(y_pred)):
    if y_test[i]==y_pred[i]:
        y_accuracy.append(1)
    else:
        y_accuracy.append(0)
print(sum(y_accuracy)/len(y_accuracy))

#交叉驗證
from xgboost import cv

params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

xgb_cv = cv(dtrain=cvdata, params=params, nfold=5,
                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)
print(xgb_cv)

#混淆矩陣
from sklearn.metrics import confusion_matrix
confusion_matix2=confusion_matrix(y_test,y_pred)
print('混淆矩陣:\n',confusion_matix2)


#%%特徵重要性
# xgb.plot_importance(model)
# plt.figure(figsize = (16, 12),dpi=800)
# plt.show()
# from xgboost import plot_tree
# from matplotlib.pylab import rcParams
# import graphviz
# graph_to_save = xgb.to_graphviz(model, num_trees = 2)
# # graph_to_save.format = 'png'            
# # graph_to_save.render('tree_2_saved')
# pd.Series(df.keys())

#%%pred now
new=pd.read_csv('C:/Users/anton/OneDrive/桌面/newpreddata2.csv')
new_x=xgb.DMatrix(new.iloc[:,1:])
newpred=model.predict(new_x,validate_features=False)
print(newpred)
# newpred=pd.DataFrame(newpred)
# newpred.to_csv('C:/Users/user/Desktop/pred.csv')
#%%儲存與叫出
#model.save_model('model_bechpredict.json')
import pandas as pd
import xgboost as xgb
new=pd.read_csv('C:/Users/anton/OneDrive/桌面/newpreddata2.csv')
new_x=xgb.DMatrix(new.iloc[:,1:])
model_xgb_2 = xgb.Booster()
model_xgb_2.load_model("model_bechpredict.json")
print(model_xgb_2.predict(new_x,validate_features=False))
