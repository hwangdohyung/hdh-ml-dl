# 데이콘 따릉이 문제풀이 
import numpy as np 
import pandas as pd # csv 파일 당겨올 때 사용
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error 
from collections import Counter
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
 
#1.데이터 
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col=0) #컬럼중에 id컬럼(0번째)은 단순 index 

test_set = pd.read_csv(path + 'test.csv', index_col=0)  #예측에서 쓴다!


#####결측치 처리 1. 제거 ######
# print(train_set.isnull().sum()) 
train_set = train_set.dropna()
# print(train_set.isnull().sum()) 
# print(train_set.shape)
test_set= test_set.fillna(test_set.mean())
##############################

x = train_set.drop(['count'], axis=1,)

y = train_set['count']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)

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
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline

# model = RandomForestClassifier()
pipe = make_pipeline(MinMaxScaler(), XGBRegressor())
model = RandomizedSearchCV(pipe,parameters,cv=5,verbose=1)

#3.훈련 
model.fit(x_train, y_train) # 파이프라인에서 fit 할땐 스케일링의 transform 과 fit이 돌아간다. 

#4.평가, 예측 
result = model.score(x_test, y_test)

print('model.r2 : ', round(result,2))

# model.r2 :  0.8



