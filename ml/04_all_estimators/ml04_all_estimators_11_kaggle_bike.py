import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import datetime as dt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv') 
            
test_set = pd.read_csv(path + 'test.csv') 


train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) 
train_set.drop('casual',axis=1,inplace=True)
train_set.drop('registered',axis=1,inplace=True)

test_set.drop('datetime',axis=1,inplace=True) 

print(train_set)
print(test_set)


x = train_set.drop(['count'], axis=1)  
print(x)
print(x.columns)
print(x.shape) # (10886, 12)

y = train_set['count'] 
print(y)
print(y.shape) # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,random_state=31)   
      
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 
test_set = scaler.transform(test_set)# **최종테스트셋이 있는경우 여기도 스케일링을 적용해야함 **               
                                                    

from sklearn.utils import all_estimators
#2.모델구성
# allAlgorithms = all_estimators(type_filter='classifier')  # 분류모델 전부를 보여준다 
allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgorithms : ', allAlgorithms)
print('모델의 갯수 : ', len(allAlgorithms))
import warnings
warnings.filterwarnings('ignore') # 출력만 안해준다

for (name, algorithm) in allAlgorithms:       # 리스트 안에 키밸류(알고리즘 이름과,위치)를 받아서 반복한다.
    try:                                      # 이것을 진행해 
        model = algorithm()
        model.fit(x_train,y_train)
    
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test,y_predict)
        print(name, '의 r2 : ', r2)              
    except:                                   # 에러가 뜨면 계속 진행해
        continue
        # print(name, '안나온놈')
        
# ARDRegression 의 r2 :  0.39839988647785085
# AdaBoostRegressor 의 r2 :  0.7177924613031355
# BaggingRegressor 의 r2 :  0.9377021779059045
# BayesianRidge 의 r2 :  0.3978595170475171
# CCA 의 r2 :  0.10813595337538151
# DecisionTreeRegressor 의 r2 :  0.8990461675875043
# DummyRegressor 의 r2 :  -0.000119201140765135
# ElasticNet 의 r2 :  0.3102454019658838
# ElasticNetCV 의 r2 :  0.39453861878257246
# ExtraTreeRegressor 의 r2 :  0.8634204569582202
# ExtraTreesRegressor 의 r2 :  0.9483471765292766
# GammaRegressor 의 r2 :  0.24442707323042034
# GaussianProcessRegressor 의 r2 :  -23.084553555569492
# GradientBoostingRegressor 의 r2 :  0.8688448846801208
# HistGradientBoostingRegressor 의 r2 :  0.9493385095171218
# HuberRegressor 의 r2 :  0.3720157547296904
# KNeighborsRegressor 의 r2 :  0.6342868695858501
# KernelRidge 의 r2 :  0.0959336359062819
# Lars 의 r2 :  0.3977037860259328
# LarsCV 의 r2 :  0.3981406017047695
# Lasso 의 r2 :  0.39846623233442124
# LassoCV 의 r2 :  0.3979482609810967
# LassoLars 의 r2 :  -0.000119201140765135
# LassoLarsCV 의 r2 :  0.3977037860259328
# LassoLarsIC 의 r2 :  0.3980020383719316
# LinearRegression 의 r2 :  0.3977037860259328
# LinearSVR 의 r2 :  0.3417084902970653
# MLPRegressor 의 r2 :  0.5971143613772676
# NuSVR 의 r2 :  0.3926467997028694
# OrthogonalMatchingPursuit 의 r2 :  0.16896908989760817
# OrthogonalMatchingPursuitCV 의 r2 :  0.3978667425786777
# PLSCanonical 의 r2 :  -0.35614008965633026
# PLSRegression 의 r2 :  0.39040983160799714
# PassiveAggressiveRegressor 의 r2 :  0.3399540805445246
# PoissonRegressor 의 r2 :  0.4015090056163173
# RANSACRegressor 의 r2 :  0.10581598215910426
# RadiusNeighborsRegressor 의 r2 :  -1.6243758610256666e+31
# RandomForestRegressor 의 r2 :  0.9465557291591497
# Ridge 의 r2 :  0.3977268699389117
# RidgeCV 의 r2 :  0.39787404766685464
# SGDRegressor 의 r2 :  0.39755632076909375
# SVR 의 r2 :  0.38729119645844556
# TheilSenRegressor 의 r2 :  0.39358593840752887
# TransformedTargetRegressor 의 r2 :  0.3977037860259328
# TweedieRegressor 의 r2 :  0.24589152107595047