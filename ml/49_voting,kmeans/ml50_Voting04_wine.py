from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
import tensorflow as tf 
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
#1. 데이터 
datasets = load_wine()

x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target, train_size=0.8,shuffle=True, random_state=123,stratify=datasets.target)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
xg = XGBClassifier(random_state=123,n_estimators =100,
              learning_rate= 0.1,
              max_depth= 3,
              gamma= 1,
              min_child_weight=1,
              subsample=1,
              colsample_bytree=1,
              colsample_bylevel=1, 
              colsample_bynode=1 ,
              reg_alpha=0,
              reg_lambda=1)
lg = LGBMClassifier()
cat = CatBoostClassifier(verbose=0)

model = VotingClassifier(estimators=[('XG', xg), ('LG', lg),('CAT',cat)],voting='soft')

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

score = accuracy_score(y_test, y_predict)
print('보팅결과 :', round(score, 4)) 

calssifiers =[xg, lg, cat]
for model2 in calssifiers:
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_predict)
    class_name = model2.__class__.__name__ 
    print('{0} 정확도 : {1:.4f}'.format(class_name, score2)) 


# 보팅결과 : 0.9722                    
# XGBClassifier 정확도 : 0.9167        
# LGBMClassifier 정확도 : 0.9722
# CatBoostClassifier 정확도 : 0.9722