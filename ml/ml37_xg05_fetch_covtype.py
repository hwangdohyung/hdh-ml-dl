from random import shuffle
from tabnanny import verbose
import numpy as np 
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
import time 

#1.데이터 
datasets = fetch_covtype()
x = datasets.data
y = datasets.target 
print(x.shape, y.shape) #(569, 30) (569,)
print(np.unique(y,return_counts=True)) #
    
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


print(y)
print(y.shape)


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=123, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 123)

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

# parameters = {'n_estimators' : [100, 200, 300, 400, 500, 1000],
#               'learning_rate': [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001],
#               'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#               'gamma': [0, 1, 2, 3, 4, 5, 7, 10, 100],
#               'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10],
#               'subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
#               'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
#               'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
#               'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] ,
#               'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10],
#               'reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10]
#               }

# https://xgboost.readthedocs.io/en/stable/parameter.html

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categories='auto',sparse= False)#False로 할 경우 넘파이 배열로 반환된다.
y = y.reshape(-1,1)
one.fit(y)
y = one.transform(y)

print(y)
print(y.shape)


#2.모델 
model = XGBClassifier(random_state = 123)

# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8,verbose=1)

model.fit(x_train,y_train)

print('최상의 매개변수 : ', model.best_params_)
print('최상의 점수 : ', model.best_score_)

results = model.score(x_test,y_test)
print(results)

