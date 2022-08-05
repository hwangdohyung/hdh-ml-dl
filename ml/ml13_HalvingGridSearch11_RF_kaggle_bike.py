import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn.experimental import enable_halving_search_cv # 실험적 버전 정식버전이 아니다 *
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV
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



x = train_set.drop(['count'], axis=1)  


y = train_set['count'] 

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)


n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle= True, random_state=66)



parameters = [
        {'n_estimators' : [100,200,300],'max_depth': [6, 8, 10, 12],'min_samples_leaf' : [3, 5, 7,10]}, # 48
        {'n_estimators' : [100,200],'max_depth': [6, 8, 10],'min_samples_split' : [2, 3, 5, 10, 12]},   # 30
        {'n_estimators' : [100,200],'max_depth': [6, 8, 10, 12],'n_jobs' : [-1, 2, 4]},                 # 32
    ]                                                                                                   # 총 합 110        

#2.모델구성
from sklearn.ensemble import RandomForestRegressor
model = RandomizedSearchCV(RandomForestRegressor(),parameters, cv =kfold, verbose=1 ,
                    refit=True, n_jobs= -1)

#3.컴파일,훈련
import time 
start = time.time()
model.fit(x_train,y_train)
end = time.time()

print("최적의 매개변수 : ", model.best_estimator_)

print("최적의 파라미터 : ", model.best_params_)

print("best_R2_ : ", model.best_score_)

print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('R2 : ', r2_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠 R2 : ', r2_score(y_test, y_pred_best))

print('걸린시간 : ', round(end - start, 2))

# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, min_samples_split=12)
# 최적의 파라미터 :  {'n_estimators': 100, 'min_samples_split': 12, 'max_depth': 10}
# best_R2_ :  0.9249850084440535
# model.score :  0.9293719096242099
# R2 :  0.9293719096242099
# 최적 튠 R2 :  0.9293719096242099
# 걸린시간 :  12.6