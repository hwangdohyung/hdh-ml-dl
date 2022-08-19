import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV,RandomizedSearchCV,StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
import datetime as dt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor

#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv') 
            
test_set = pd.read_csv(path + 'test.csv') 


train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) 
train_set.drop('casual',axis=1,inplace=True)
train_set.drop('registered',axis=1,inplace=True)

test_set.drop('datetime',axis=1,inplace=True) 



x = train_set.drop(['count'], axis=1)  
x = train_set.drop(['count'], axis=1)  


y = train_set['count'] 


from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer,PowerTransformer
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True,random_state=1234)

scaler_li = [StandardScaler(),MinMaxScaler(),MaxAbsScaler(),RobustScaler(),QuantileTransformer(),PowerTransformer(method='yeo-johnson')]
model_li = [LinearRegression(),XGBRegressor()]

for i,v in scaler_li,model_li:
    scaler = i
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = v
    model.fit(x_train,y_train)

    y_predict = model.predict(x_test)
    print('결과 : ',round(r2_score(y_test,y_predict),4))


# scaler = PowerTransformer(method='yeo-johnson')  # 디폴트    
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = PowerTransformer(method='box-cox')
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
