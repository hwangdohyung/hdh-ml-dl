import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn.model_selection import train_test_split,KFold,cross_val_score,cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
import datetime as dt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

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

print(train_set)
print(test_set)


x = train_set.drop(['count'], axis=1)  
print(x)
print(x.columns)
print(x.shape) # (10886, 12)

y = train_set['count'] 
print(y)
print(y.shape) # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,random_state=31)   
      
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 
test_set = scaler.transform(test_set)# **최종테스트셋이 있는경우 여기도 스케일링을 적용해야함 **               
                                                    

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle= True, random_state=66)  # 5개 중에 1 개를 val 로 쓰겠다. 교차검증!

#2.모델구성
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

#3.4.컴파일,훈련,평가,예측

scores = cross_val_score(model, x_train, y_train, cv= kfold)
print('R2 : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
r2 = r2_score(y_test,y_predict)
print('cross_val_predict r2 : ', r2)

# R2 :  [0.72025027 0.77547057 0.75560377 0.80173517 0.72845195] 
#  cross_val_score :  0.7563
# cross_val_predict r2 :  0.7234343384097666

# R2 :  [0.94899319 0.93709063 0.94912714 0.95452264 0.94390002] 
#  cross_val_score :  0.9467
# cross_val_predict r2 :  0.8967444287653461