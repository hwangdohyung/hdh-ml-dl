from random import shuffle
import numpy as np 
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
import time 

#1.데이터 
datasets = load_digits()
x = datasets.data
y = datasets.target 
print(x.shape, y.shape) #(569, 30) (569,)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=123, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 123)
from sklearn.ensemble import BaggingClassifier
#2.모델 
model =BaggingClassifier(XGBClassifier(random_state = 123,n_estimators =200,
              learning_rate= 0.1,
              max_depth= 3,
              gamma= 0,
              min_child_weight=1,
              subsample=0.7,
              colsample_bytree=0.3,
              colsample_bylevel=0.5, 
              colsample_bynode=1 ,
              reg_alpha=0.001,
              reg_lambda=0.001))

# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

model.fit(x_train,y_train)

# print('최상의 매개변수 : ', model.best_params_)
# print('최상의 점수 : ', model.best_score_)

results = model.score(x_test,y_test)
print(results)

# 0.9861111111111112