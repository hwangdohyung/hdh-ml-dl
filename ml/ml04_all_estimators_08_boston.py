from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler


#1.데이터
datasets = load_boston()

x = datasets['data']
y = datasets['target']


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)

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

# ARDRegression 의 r2 :  0.729221134484114
# AdaBoostRegressor 의 r2 :  0.8056617069222092
# BaggingRegressor 의 r2 :  0.7972016240495218
# BayesianRidge 의 r2 :  0.7485312679074179
# CCA 의 r2 :  0.6715680493231329
# DecisionTreeRegressor 의 r2 :  0.6874897902288662
# DummyRegressor 의 r2 :  -0.02483494500221739
# ElasticNet 의 r2 :  0.738480403933147
# ElasticNetCV 의 r2 :  0.7313489300084908
# ExtraTreeRegressor 의 r2 :  0.7030602318877002
# ExtraTreesRegressor 의 r2 :  0.8721315866981508
# GammaRegressor 의 r2 :  -0.02483494500221739
# GaussianProcessRegressor 의 r2 :  -5.533995812823804
# GradientBoostingRegressor 의 r2 :  0.8326198406200152
# HistGradientBoostingRegressor 의 r2 :  0.8630314964335115
# HuberRegressor 의 r2 :  0.7191530549836833
# KNeighborsRegressor 의 r2 :  0.4823983034178225
# KernelRidge 의 r2 :  0.7088995973401755
# Lars 의 r2 :  0.7567743451802433
# LarsCV 의 r2 :  0.7562421448832327
# Lasso 의 r2 :  0.7388953630087362
# LassoCV 의 r2 :  0.7484324657984536
# LassoLars 의 r2 :  -0.02483494500221739
# LassoLarsCV 의 r2 :  0.7558097153816461
# LassoLarsIC 의 r2 :  0.7275899672282478
# LinearRegression 의 r2 :  0.7567743451802437
# LinearSVR 의 r2 :  0.5433658555426724
# MLPRegressor 의 r2 :  0.7146785489529986
# NuSVR 의 r2 :  0.3107923309977306
# OrthogonalMatchingPursuit 의 r2 :  0.3809706188269514
# OrthogonalMatchingPursuitCV 의 r2 :  0.7089262666740079
# PLSCanonical 의 r2 :  -2.1102728653947493
# PLSRegression 의 r2 :  0.7224577486131363
# PassiveAggressiveRegressor 의 r2 :  -0.0785613887751837
# PoissonRegressor 의 r2 :  0.8006698592753809
# RANSACRegressor 의 r2 :  0.607814841208721
# RandomForestRegressor 의 r2 :  0.8386625671445382
# Ridge 의 r2 :  0.7512895046791443
# RidgeCV 의 r2 :  0.7559702798619649
# SGDRegressor 의 r2 :  -9.17720375779636e+25
# SVR 의 r2 :  0.3033944740435023
# TheilSenRegressor 의 r2 :  0.7017806579115446
# TransformedTargetRegressor 의 r2 :  0.7567743451802437
# TweedieRegressor 의 r2 :  0.7238951799833455