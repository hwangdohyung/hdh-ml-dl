#[실습]
from random import shuffle
import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
import time 

#1.데이터 
datasets = load_diabetes()
x = datasets.data
y = datasets.target 
print(x.shape, y.shape) 

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=123)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state = 123)

# 'n_estimators' : [100, 200, 300, 400, 500, 1000] # 디폴트 100 / 1~inf  (inf: 무한대)
# 'learning_rate': [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] 디폴트 0.3/ 0~1 / eta라고 써도 먹힘
# 'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] 디폴트 6 / 0~ inf / 정수
# 'gamma': [0, 1, 2, 3, 4, 5, 7, 10, 100] 디폴트 0/ 0~inf
# 'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10] 디폴트 1 / 0~inf
# 'subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 0/ 0~inf / L1 절대값 가중치 규제 /alpha
# 'reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 1/ 0~inf/ L2 제곱 가중치 규제 /lambda

parameters = {'n_estimators' : [100],
              'learning_rate': [0.2],
              'max_depth': [2],
              'gamma': [100],
              'min_child_weight': [10],
              'subsample': [0.1],
              'colsample_bytree': [0],
              'colsample_bylevel': [0],
              'colsample_bynode': [0] ,
              'reg_alpha': [2],
              'reg_lambda':[10]
              }

# https://xgboost.readthedocs.io/en/stable/parameter.html

#2.모델 
xgb = XGBRegressor(random_state = 123)

model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

model.fit(x_train,y_train)

print('최상의 매개변수 : ', model.best_params_)
print('최상의 점수 : ', model.best_score_)

# 최상의 매개변수 :  {'n_estimators': 100}
# 최상의 점수 :  0.1985136084431129

# 최상의 매개변수 :  {'learning_rate': 0.2, 'n_estimators': 100}
# 최상의 점수 :  0.2570991637493167

# 최상의 매개변수 :  {'learning_rate': 0.2, 'max_depth': 2, 'n_estimators': 100}
# 최상의 점수 :  0.3250976232202877

# 최상의 매개변수 :  {'gamma': 100, 'learning_rate': 0.2, 'max_depth': 2, 'n_estimators': 100}
# 최상의 점수 :  0.32846246835845916

# 최상의 매개변수 :  {'gamma': 100, 'learning_rate': 0.2, 'max_depth': 2, 'min_child_weight': 10, 'n_estimators': 100}
# 최상의 점수 :  0.34121016853334

# 최상의 매개변수 :  {'gamma': 100, 'learning_rate': 0.2, 'max_depth': 2, 'min_child_weight': 10, 'n_estimators': 100, 'subsample': 0.1}
# 최상의 점수 :  0.36482585307346005

# 최상의 매개변수 :  {'colsample_bytree': 0, 'gamma': 100, 'learning_rate': 0.2, 'max_depth': 2, 'min_child_weight': 10, 'n_estimators': 100, 'subsample': 0.1}
# 최상의 점수 :  0.39483906939234714

# 최상의 매개변수 :  {'colsample_bylevel': 0, 'colsample_bytree': 0, 'gamma': 100, 'learning_rate': 0.2, 'max_depth': 2, 'min_child_weight': 10, 'n_estimators': 100, 'subsample': 0.1}
# 최상의 점수 :  0.39483906939234714

# 최상의 매개변수 :  {'colsample_bylevel': 0, 'colsample_bynode': 0, 'colsample_bytree': 0, 'gamma': 100, 'learning_rate': 0.2, 'max_depth': 2, 'min_child_weight': 10, 'n_estimators': 100, 'subsample': 0.1}
# 최상의 점수 :  0.39483906939234714

# 최상의 매개변수 :  {'colsample_bylevel': 0, 'colsample_bynode': 0, 'colsample_bytree': 0, 'gamma': 100, 'learning_rate': 0.2, 'max_depth': 2, 'min_child_weight': 10, 'n_estimators': 100, 'reg_alpha': 2, 'subsample': 0.1}
# 최상의 점수 :  0.39571911431704077

# 최상의 매개변수 :  {'colsample_bylevel': 0, 'colsample_bynode': 0, 'colsample_bytree': 0, 'gamma': 100, 'learning_rate': 0.2, 'max_depth': 2, 'min_child_weight': 10, 'n_estimators': 100, 'reg_alpha': 2, 'reg_lambda': 10, 'subsample': 0.1}
# 최상의 점수 :  0.3957456930264397

results = model.score(x_test,y_test)
print(results)

# 0.44976941768853285