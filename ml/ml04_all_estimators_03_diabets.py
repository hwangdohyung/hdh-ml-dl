from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler


#1.데이터
datasets = load_diabetes()

x = datasets['data']
y = datasets['target']
print(x.shape, y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)


scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 

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
        
# ARDRegression 의 r2 :  0.5715276342376795
# AdaBoostRegressor 의 r2 :  0.5793972421858287
# BaggingRegressor 의 r2 :  0.5328272071396312
# BayesianRidge 의 r2 :  0.5728134204050608
# CCA 의 r2 :  0.5485329309725473
# DecisionTreeRegressor 의 r2 :  0.15322734670234783
# DummyRegressor 의 r2 :  -5.1050706625588305e-05
# ElasticNet 의 r2 :  0.4848908256787521
# ElasticNetCV 의 r2 :  0.5698626895328331
# ExtraTreeRegressor 의 r2 :  -0.005426305664394393
# ExtraTreesRegressor 의 r2 :  0.5466329822452558
# GammaRegressor 의 r2 :  0.41821229779268354
# GaussianProcessRegressor 의 r2 :  -0.006352988477490218
# GradientBoostingRegressor 의 r2 :  0.5697566913448306
# HistGradientBoostingRegressor 의 r2 :  0.553958090092293
# HuberRegressor 의 r2 :  0.5597169506631436
# KNeighborsRegressor 의 r2 :  0.45600480416443656
# KernelRidge 의 r2 :  -0.8875567073973123
# Lars 의 r2 :  0.5648381389215891
# LarsCV 의 r2 :  0.5283635469872618
# Lasso 의 r2 :  0.5692247267146126
# LassoCV 의 r2 :  0.5665840117243639
# LassoLars 의 r2 :  0.40713993611532073
# LassoLarsCV 의 r2 :  0.5648381389215893
# LassoLarsIC 의 r2 :  0.5720443202386556
# LinearRegression 의 r2 :  0.5648381389215897
# LinearSVR 의 r2 :  0.2791930823091435
# MLPRegressor 의 r2 :  -0.9818084784511394
# NuSVR 의 r2 :  0.1686633291123093
# OrthogonalMatchingPursuit 의 r2 :  0.35316494914469476
# OrthogonalMatchingPursuitCV 의 r2 :  0.5531819669501177
# PLSCanonical 의 r2 :  -0.6581094498300968
# PLSRegression 의 r2 :  0.5702684686229571
# PassiveAggressiveRegressor 의 r2 :  0.548964569449732
# PoissonRegressor 의 r2 :  0.5829865747742935
# RANSACRegressor 의 r2 :  0.1440023896478937
# RandomForestRegressor 의 r2 :  0.5705158593750383
# Ridge 의 r2 :  0.5705045594673042
# RidgeCV 의 r2 :  0.5658714042508168
# SGDRegressor 의 r2 :  0.5752250845215781
# SVR 의 r2 :  0.15993670767654755
# TheilSenRegressor 의 r2 :  0.5607794386775835
# TransformedTargetRegressor 의 r2 :  0.5648381389215897
# TweedieRegressor 의 r2 :  0.42071255109755834