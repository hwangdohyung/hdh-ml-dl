from random import shuffle
import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
import time 
from sklearn.metrics import accuracy_score,r2_score

#1.데이터 
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target 
print(x.shape, y.shape) 

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=123, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 123)


parameters = {'n_estimators' : [100],'learning_rate': [0.1],'max_depth': [3],'gamma': [1],
              'min_child_weight': [1],'subsample': [1],'colsample_bytree': [1],'colsample_bylevel': [1],
              'colsample_bynode': [1] ,'reg_alpha': [0],'reg_lambda':[1]
              }

#2.모델 
model = XGBClassifier(random_state = 123,
                      n_estimators = 1000,
                      learning_rate = 0.1,
                      max_depth = 3,
                      gamma = 1, 
                      )

# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

model.fit(x_train,y_train,early_stopping_rounds=10,
            eval_set =[(x_train,y_train),(x_test,y_test)],
        #   eval_set =[(x_test,y_test)]
            eval_metric = 'error'
                # 회귀: rmse, mae, rmsle ... 등등
                # 이진: error, logloss, auc... 등
                # 다중: merror, mlogloss... 등 
          )  # 10번동안 갱신 없으면 정지 시키겠다

# print('최상의 매개변수 : ', model.best_params_)
# print('최상의 점수 : ', model.best_score_)

results = model.score(x_test,y_test)
print(results)

print('===========================')
hist = model.evals_result()
print(hist)

#[실습]
# 그래프 그려봐 !!
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False
#######################################################

# plt.figure(figsize=(9,6)) #그래프 표 사이즈
# plt.plot(hist.history['validation_0'], marker = '.' ,c = 'red', label = 'val0') # maker: 점으로 표시하겟다  ,c:색깔 ,label : 이름
# plt.plot(hist.history['validation_1'], marker = '.' ,c = 'blue', label = 'val1')
# plt.grid() # 모눈종이에 하겠다
# plt.title('eval')#제목
# plt.ylabel('val0')#y축 이름
# plt.xlabel('val1')#x축 이름
# plt.legend(loc='upper right') # upper right: 위쪽 상단에 표시하겠다.(라벨 이름들)
# plt.show()# 보여줘

 
# plt.plot(hist.values(), '-', color='C0', label='Solid')
# plt.plot(hist, '-.', color='C0', label='Dash-dot')
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.axis([0.8, 3.2, 0.5, 5.0])
# plt.legend(loc='upper right', ncol=4)
# plt.show()

# plt.show()

print('==================')
a = hist['validation_0'].values()
b = hist['validaiont_1'].values()
print(a)
print(b)
