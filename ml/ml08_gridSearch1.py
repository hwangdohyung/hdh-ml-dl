from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold

#1.데이터
datasets = load_iris()

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
from sklearn.linear_model import Perceptron,LogisticRegression  #logisticregression : regression 이 들어가지만 분류다!
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = SVC(c=1, kernel='linear', degree=3)
model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1,
                     refit=True, n_jobs= -1)                   #n_jobs : cpu 갯수 n 쓰겟다 (-1 전체 다 쓰겟다.)        # 42 * 5 = 210번(kofld포함)

    
#3.컴파일,훈련
import time 
start = time.time()
model.fit(x_train,y_train)
end = time.time()

print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear')

print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

print("best_score_ : ", model.best_score_)
# best_score_ :  0.980952380952381

print('model.score : ', model.score(x_test, y_test))
# model.score :  0.9555555555555556

y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))
# accuracy_score :  0.9555555555555556

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))
# 최적 튠 ACC :  0.9555555555555556

print('걸린시간 : ', round(end - start, 2))
# 걸린시간 :  1.26