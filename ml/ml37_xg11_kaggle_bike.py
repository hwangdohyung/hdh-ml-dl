import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV,RandomizedSearchCV,StratifiedKFold
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
kfold = StratifiedKFold(n_splits=n_splits, shuffle= True, random_state=66)

parameters = {'n_estimators' : [100, 200, 300, 400, 500, 1000],
              'learning_rate': [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001],
              'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'gamma': [0, 1, 2, 3, 4, 5, 7, 10, 100],
              'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10],
              'subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
              'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
              'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
              'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] ,
              'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10],
              'reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10]
              }

#2. 모델 
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline 
from xgboost import XGBRegressor
pipe = Pipeline([('minmax', MinMaxScaler()), ('RF', XGBRegressor())],verbose=1) # '' 는 그냥 변수명 (파이프라인에 그리드서치 엮을때 모델명 변수를 파라미터에 명시해 줘야 됨.)


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
# model = GridSearchCV(RandomForestClassifier(), parameters, cv=5, verbose=1)

model.fit(x_train, y_train) # 파이프라인에서 fit 할땐 스케일링의 transform 과 fit이 돌아간다. 

#4.평가, 예측 
result = model.score(x_test, y_test)

print('model.score : ', round(result,2))

# model.score :  0.94