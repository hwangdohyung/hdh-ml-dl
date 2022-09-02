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


#2. 모델 
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,BaggingRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline

model =BaggingRegressor (XGBRegressor(random_state=123,
                                     n_estimators=100,
                                     learning_rate=0.2,
                                     max_depth=9,
                                     gamma=100,
                                     min_child_weight=0.1,
                                     subsample=1,
                                     colsample_bytree=1,
                                     colsample_byleve=0.3,
                                     colsample_bynode=0.7 ,
                                     reg_alpha=0.01,
                                     reg_lambda=0.1))

# model = RandomForestClassifier()
# pipe = make_pipeline(MinMaxScaler(), XGBRegressor())
# model = RandomizedSearchCV(xgb,parameters,cv=5,verbose=1)

#3.훈련 
model.fit(x_train, y_train) # 파이프라인에서 fit 할땐 스케일링의 transform 과 fit이 돌아간다. 

#4.평가, 예측 
result = model.score(x_test, y_test)

# print('최상의 매개변수 : ', model.best_params_)
# print('최상의 점수 : ', model.best_score_)


print('model.r2 : ', round(result,2))

# model.r2 :  0.8
# model.r2 :  0.8


