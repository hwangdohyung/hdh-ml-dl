

from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split,RandomizedSearchCV
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

from sklearn.model_selection import train_test_split,KFold,cross_val_score

#1.데이터
datasets = load_breast_cancer()
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
from sklearn.ensemble import RandomForestClassifier
model = RandomizedSearchCV(RandomForestClassifier(),parameters, cv =kfold, verbose=1 ,
                    refit=True, n_jobs= -1)

#3.컴파일,훈련
import time 
start = time.time()
model.fit(x_train,y_train)
end = time.time()

print("최적의 매개변수 : ", model.best_estimator_)

print("최적의 파라미터 : ", model.best_params_)

print("best_score_ : ", model.best_score_)

print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))

print('걸린시간 : ', round(end - start, 2))
## grid ##0
# 최적의 매개변수 :  RandomForestClassifier(max_depth=10, min_samples_split=3, n_estimators=200)
# 최적의 파라미터 :  {'max_depth': 10, 'min_samples_split': 3, 'n_estimators': 200}
# best_score_ :  0.9648417721518987
# model.score :  0.9532163742690059
# accuracy_score :  0.9532163742690059
# 최적 튠 ACC :  0.9532163742690059
# 걸린시간 :  15.34

### random ###
# 최적의 매개변수 :  RandomForestClassifier(max_depth=12, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'n_estimators': 100, 'max_depth': 12}
# best_score_ :  0.9623417721518986
# model.score :  0.9590643274853801
# accuracy_score :  0.9590643274853801
# 최적 튠 ACC :  0.9590643274853801
# 걸린시간 :  2.45