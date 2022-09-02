# 데이콘 따릉이 문제풀이 
import numpy as np 
import pandas as pd # csv 파일 당겨올 때 사용
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense,Dropout
from sklearn.experimental import enable_halving_search_cv # 실험적 버전 정식버전이 아니다 *
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV
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

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.9,random_state=31)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle= True, random_state=66)



parameters = [
        {'n_estimators' : [100,200,300],'max_depth': [6, 8, 10, 12],'min_samples_leaf' : [3, 5, 7,10]}, # 48
        {'n_estimators' : [100,200],'max_depth': [6, 8, 10],'min_samples_split' : [2, 3, 5, 10, 12]},   # 30
        {'n_estimators' : [100,200],'max_depth': [6, 8, 10, 12],'n_jobs' : [-1, 2, 4]},                 # 32
    ]                                                                                                   # 총 합 110        

#2.모델구성
from sklearn.ensemble import RandomForestRegressor
model = HalvingGridSearchCV(RandomForestRegressor(),parameters, cv =kfold, verbose=1 ,
                    refit=True, n_jobs= -1)

#3.컴파일,훈련
import time 
start = time.time()
model.fit(x_train,y_train)
end = time.time()

print("최적의 매개변수 : ", model.best_estimator_)

print("최적의 파라미터 : ", model.best_params_)

print("best_R2_ : ", model.best_score_)

print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('R2 : ', r2_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠 R2 : ', r2_score(y_test, y_pred_best))

print('걸린시간 : ', round(end - start, 2))

# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, min_samples_split=10)
# 최적의 파라미터 :  {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 100}
# best_R2_ :  0.7371378456979105
# model.score :  0.7909893699334173
# R2 :  0.7909893699334173
# 최적 튠 R2 :  0.7909893699334173
# 걸린시간 :  18.84