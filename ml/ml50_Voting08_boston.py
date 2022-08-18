from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier,VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score,r2_score
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
import tensorflow as tf 
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from xgboost import XGBClassifier,XGBRegressor

#1. 데이터 
datasets = load_boston()

x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target, train_size=0.8,shuffle=True, random_state=123)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
xg = XGBRegressor(random_state = 123,n_estimators =100,
              learning_rate= 0.2,
              max_depth= 2 ,
              gamma= 100,
              min_child_weight=10,
              subsample=0.1,
              colsample_bytree=0,
              colsample_bylevel=0,
              colsample_bynode=0,
              reg_alpha=2,
              reg_lambda=10)
lg = LGBMRegressor()
cat = CatBoostRegressor(verbose=0)

model = VotingRegressor(estimators=[('XG', xg), ('LG', lg),('CAT',cat)])

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

score = r2_score(y_test, y_predict)
print('보팅결과 :', round(score, 4)) 

calssifiers =[xg, lg, cat]
for model2 in calssifiers:
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test, y_predict)
    class_name = model2.__class__.__name__ 
    print('{0} 정확도 : {1:.4f}'.format(class_name, score2)) 
    
# 보팅결과 : 0.8143
# XGBRegressor 정확도 : 0.5585
# LGBMRegressor 정확도 : 0.7481
# CatBoostRegressor 정확도 : 0.8763