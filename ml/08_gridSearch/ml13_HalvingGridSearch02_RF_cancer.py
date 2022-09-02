from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold

from sklearn.experimental import enable_halving_search_cv # 실험적 버전 정식버전이 아니다 *
from sklearn.model_selection import HalvingGridSearchCV

#1.데이터
datasets = load_breast_cancer()

x = datasets['data']
y = datasets['target']


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
              {"C":[1, 10, 100, 1000],"kernel":["linear"], "degree":[3,4,5]},   #12번
              {"C":[1, 10, 100],"kernel":["rbf"], "gamma":[0.001,0.0001]},      #6번
              {"C":[1, 10, 100, 1000],"kernel":["sigmoid"],
               "gamma":[0.01, 0.001, 0.0001], "degree":[3,4]}]                  #24번
                                                                                #파라미터 설정 총 42번


#2.모델구성
from sklearn.svm import LinearSVC,SVC

model = HalvingGridSearchCV(SVC(), parameters, cv=kfold, verbose=1,
                     refit=True, n_jobs= -1)                   #n_jobs : cpu 갯수 n 쓰겟다 (-1 전체 다 쓰겟다.)        # 42 * 5 = 210번(kofld포함)

    
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

# n_iterations: 3
# n_required_iterations: 4
# n_possible_iterations: 3
# min_resources_: 20
# max_resources_: 398
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 42
# n_resources: 20
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# ----------
# iter: 1
# n_candidates: 14
# n_resources: 60
# Fitting 5 folds for each of 14 candidates, totalling 70 fits
# ----------
# iter: 2
# n_candidates: 5
# n_resources: 180
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# 최적의 매개변수 :  SVC(C=100, degree=5, kernel='linear')
# 최적의 파라미터 :  {'C': 100, 'degree': 5, 'kernel': 'linear'}
# best_score_ :  0.9382539682539683
# model.score :  0.9415204678362573
# accuracy_score :  0.9415204678362573
# 최적 튠 ACC :  0.9415204678362573
# 걸린시간 :  12.11