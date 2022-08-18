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
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
 
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

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.8,random_state=31)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
xg = XGBRegressor(random_state=123,
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
                                     reg_lambda=0.1)
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


# XGBRegressor 정확도 : 0.7875
# LGBMRegressor 정확도 : 0.7893
# CatBoostRegressor 정확도 : 0.8174