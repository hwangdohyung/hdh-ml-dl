
from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, train_test_split,RandomizedSearchCV
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

from sklearn.model_selection import train_test_split,KFold,cross_val_score

#1.데이터
datasets = load_digits()
x = datasets['data']
y = datasets['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle= True, random_state=66)  # 5개 중에 1 개를 val 로 쓰겠다. 교차검증!


parameters = [
        {'n_estimators' : [100,200,300],'max_depth': [6, 8, 10, 12],'min_samples_leaf' : [3, 5, 7,10]}, # 48
        {'n_estimators' : [100,200],'max_depth': [6, 8, 10],'min_samples_split' : [2, 3, 5, 10, 12]},   # 30
        {'n_estimators' : [100,200],'max_depth': [6, 8, 10, 12],'n_jobs' : [-1, 2, 4]},                 # 32
    ]                                                                                                   # 총 합 110        

#2.모델구성
from sklearn.ensemble import RandomForestRegressor
model = RandomizedSearchCV(RandomForestRegressor(),parameters, cv =kfold, verbose=1 ,
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
## grid ##
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10)
# 최적의 파라미터 :  {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 100}
# best_R2_ :  0.8343077508606018
# model.score :  0.8397937538655744
# R2 :  0.8397937538655744
# 최적 튠 R2 :  0.8397937538655744
# 걸린시간 :  67.65

## random ##
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'n_estimators': 100, 'max_depth': 10}
# best_R2_ :  0.830660436415877
# model.score :  0.8395072406362609
# R2 :  0.8395072406362609
# 최적 튠 R2 :  0.8395072406362609
# 걸린시간 :  7.41