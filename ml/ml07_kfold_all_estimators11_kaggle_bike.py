import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn.model_selection import train_test_split,KFold,cross_val_score
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



x = train_set.drop(['count'], axis=1)  


y = train_set['count'] 


n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle= True, random_state=66)



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
  
        r2 = cross_val_score(model, x,y , cv=kfold)
        print(name, '의 r2 : ', r2)   
                   
    except:                                   # 에러가 뜨면 계속 진행해
        continue
        # print(name, '안나온놈')

# ARDRegression 의 r2 :  [0.37370221 0.41197783 0.38154727 0.38157259 0.39108942]
# AdaBoostRegressor 의 r2 :  [0.67662941 0.69948629 0.65948454 0.68892486 0.63577371]
# BaggingRegressor 의 r2 :  [0.94174285 0.94350265 0.94533229 0.95021055 0.93831406]
# BayesianRidge 의 r2 :  [0.37315803 0.41162025 0.38183776 0.38156147 0.39141605]
# CCA 의 r2 :  [0.05149681 0.16351092 0.05990954 0.02820278 0.08763834]
# DecisionTreeRegressor 의 r2 :  [0.90534599 0.90066756 0.89391981 0.91553673 0.88774046]
# DummyRegressor 의 r2 :  [-6.49419743e-04 -4.44297774e-04 -2.40830679e-05 -1.65108679e-03
#  -5.93256979e-08]
# ElasticNet 의 r2 :  [0.35142773 0.38158674 0.36098013 0.36012872 0.36732731]
# ElasticNetCV 의 r2 :  [0.33995661 0.36868    0.34969005 0.3490424  0.35529731]
# ExtraTreeRegressor 의 r2 :  [0.88952515 0.87494612 0.87394288 0.88989716 0.84655655]
# ExtraTreesRegressor 의 r2 :  [0.94966195 0.95191294 0.94894041 0.95906666 0.9494896 ]
# GammaRegressor 의 r2 :  [0.28638159 0.30715047 0.28736147 0.27999711 0.30711832]
# GaussianProcessRegressor 의 r2 :  [-0.81443653 -0.77982962 -0.80118165 -0.84785518 -0.75799043]
# GradientBoostingRegressor 의 r2 :  [0.84855364 0.86424046 0.85986187 0.86650171 0.86071198]
# HistGradientBoostingRegressor 의 r2 :  [0.95357944 0.95280724 0.95119486 0.9573275  0.95158478]
# HuberRegressor 의 r2 :  [0.35269582 0.37497876 0.35254691 0.34201604 0.35995105]
# IsotonicRegression 의 r2 :  [nan nan nan nan nan]
# KNeighborsRegressor 의 r2 :  [0.57205624 0.58122617 0.60526882 0.57932249 0.57047332]
# KernelRidge 의 r2 :  [0.3729483  0.41169654 0.38194265 0.38205462 0.39130052]
# Lars 의 r2 :  [0.37304195 0.41178093 0.38121844 0.38153711 0.3914349 ]
# LarsCV 의 r2 :  [0.3743054  0.40212342 0.36269373 0.38146185 0.38974002]
# Lasso 의 r2 :  [0.37387918 0.41160684 0.38139421 0.38109327 0.39110133]
# LassoCV 의 r2 :  [0.37389347 0.4115083  0.38135354 0.38107469 0.39107491]
# LassoLars 의 r2 :  [-6.49419743e-04 -4.44297774e-04 -2.40830679e-05 -1.65108679e-03
#  -5.93256979e-08]
# LassoLarsCV 의 r2 :  [0.37429779 0.41102235 0.38163952 0.38146594 0.39061299]
# LassoLarsIC 의 r2 :  [0.37422508 0.4118174  0.38162198 0.38140934 0.39076241]
# LinearRegression 의 r2 :  [0.37304195 0.41178093 0.38175646 0.38153711 0.3914349 ]
# LinearSVR 의 r2 :  [0.33943251 0.36477252 0.3329529  0.31671853 0.33782317]
# MLPRegressor 의 r2 :  [0.55167696 0.58294863 0.58023689 0.56261611 0.55359214]
# MultiTaskElasticNet 의 r2 :  [nan nan nan nan nan]
# MultiTaskElasticNetCV 의 r2 :  [nan nan nan nan nan]
# MultiTaskLasso 의 r2 :  [nan nan nan nan nan]
# MultiTaskLassoCV 의 r2 :  [nan nan nan nan nan]
# NuSVR 의 r2 :  [0.27543625 0.284376   0.27280372 0.26454685 0.2744837 ]
# OrthogonalMatchingPursuit 의 r2 :  [0.15590607 0.16800949 0.15578748 0.16026013 0.16007406]
# OrthogonalMatchingPursuitCV 의 r2 :  [0.37276203 0.41100042 0.38000557 0.37853882 0.3899379 ]
# PLSCanonical 의 r2 :  [-0.40667397 -0.25114062 -0.34788324 -0.34953621 -0.28741366]
# PLSRegression 의 r2 :  [0.36909413 0.40743944 0.3768716  0.37406291 0.38410672]
# PassiveAggressiveRegressor 의 r2 :  [-0.12226478  0.1540239  -0.3652766   0.35400246  0.20991796]
# PoissonRegressor 의 r2 :  [0.37826856 0.43610443 0.40455782 0.40457851 0.41795195]
# RANSACRegressor 의 r2 :  [0.28298275 0.03226213 0.028988   0.02302858 0.22053005]
# RadiusNeighborsRegressor 의 r2 :  [-2.36644833e+33 -2.27266132e+33 -2.29580634e+33 -2.25178462e+33
#  -2.26568952e+33]
# RandomForestRegressor 의 r2 :  [0.94691623 0.94983567 0.94747292 0.95849265 0.94723064]
# Ridge 의 r2 :  [0.3730461  0.41177683 0.38175943 0.38153819 0.39143472]
# RidgeCV 의 r2 :  [0.37308156 0.4117387  0.38178462 0.38154693 0.39143205]
# SGDRegressor 의 r2 :  [-7.32023481e+11 -3.85921582e+14 -5.32645074e+13 -1.67695583e+13
#  -1.40226247e+14]
# SVR 의 r2 :  [0.259242   0.27179279 0.25894131 0.24068804 0.26202432]
# TheilSenRegressor 의 r2 :  [0.37113537 0.40474475 0.37620933 0.37529286 0.3877256 ]
# TransformedTargetRegressor 의 r2 :  [0.37304195 0.41178093 0.38175646 0.38153711 0.3914349 ]
# TweedieRegressor 의 r2 :  [0.34221446 0.37115998 0.35040777 0.35097608 0.3579819 ]