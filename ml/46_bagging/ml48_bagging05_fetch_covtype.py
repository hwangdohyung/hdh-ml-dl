from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import VotingClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
import tensorflow as tf 
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

#1. 데이터 
datasets = fetch_covtype()
x = datasets.data 
y = datasets.target

le = LabelEncoder()
y = le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=123,shuffle=True,stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2.모델
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
# from sklearn.linear_model import LogisticRegression
model = BaggingClassifier(DecisionTreeClassifier(),     # randomforest와 같다고 볼수있다. 
                          n_estimators = 100,
                          n_jobs=-1,
                          random_state=123,
                           )
#3.훈련
model.fit(x_train,y_train)


#4.평가,예측
print('결과: ',model.score(x_test,y_test)) 


# 결과:  0.9679440289837612

