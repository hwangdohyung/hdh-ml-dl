from random import shuffle
import numpy as np 
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
import time 

#1.데이터 
datasets = load_boston()
x = datasets.data
y = datasets.target 
print(x.shape, y.shape) #(569, 30) (569,)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=123, 
                                                #  stratify=y
                                                )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 123)
from sklearn.ensemble import BaggingClassifier,BaggingRegressor
#2.모델 
model =BaggingRegressor(XGBRegressor(random_state = 123,n_estimators =100,
              learning_rate= 0.2,
              max_depth= 2 ,
              gamma= 100,
              min_child_weight=10,
              subsample=0.1,
              colsample_bytree=0,
              colsample_bylevel=0,
              colsample_bynode=0,
              reg_alpha=2,
              reg_lambda=10))

# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

model.fit(x_train,y_train)

# print('최상의 매개변수 : ', model.best_params_)
# print('최상의 점수 : ', model.best_score_)

results = model.score(x_test,y_test)
print(results)

# 0.5625029574955622

# 0.5594175712371957
