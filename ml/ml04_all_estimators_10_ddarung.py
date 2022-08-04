# 데이콘 따릉이 문제풀이 
import numpy as np 
import pandas as pd # csv 파일 당겨올 때 사용
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
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

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=48)


scaler = MaxAbsScaler()

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
        
# ARDRegression 의 r2 :  0.5673537894412475
# AdaBoostRegressor 의 r2 :  0.5916997761094008
# BaggingRegressor 의 r2 :  0.7457531511041902
# BayesianRidge 의 r2 :  0.5670973304004406
# CCA 의 r2 :  0.2661525735115735
# DecisionTreeRegressor 의 r2 :  0.6378269101748807
# DummyRegressor 의 r2 :  -0.000413564213229467
# ElasticNet 의 r2 :  0.18519956650135572
# ElasticNetCV 의 r2 :  0.541623172972185
# ExtraTreeRegressor 의 r2 :  0.5650304230844615
# ExtraTreesRegressor 의 r2 :  0.7983449077604955
# GammaRegressor 의 r2 :  0.1106762525927677
# GaussianProcessRegressor 의 r2 :  -22.632433886172727
# GradientBoostingRegressor 의 r2 :  0.7683690284357507
# HistGradientBoostingRegressor 의 r2 :  0.7811946060559773
# HuberRegressor 의 r2 :  0.5528879947396701
# KNeighborsRegressor 의 r2 :  0.7216798128205444
# KernelRidge 의 r2 :  0.5640980971884741
# Lars 의 r2 :  0.5677083527727917
# LarsCV 의 r2 :  0.5677083527727917
# Lasso 의 r2 :  0.5462758931560012
# LassoCV 의 r2 :  0.5676135947573981
# LassoLars 의 r2 :  0.3214021099722246
# LassoLarsCV 의 r2 :  0.5677083527727917
# LassoLarsIC 의 r2 :  0.5669565802185315
# LinearRegression 의 r2 :  0.567708352772792
# LinearSVR 의 r2 :  0.4189774601478101
# MLPRegressor 의 r2 :  0.4270992889443258
# NuSVR 의 r2 :  0.4253467957900803
# OrthogonalMatchingPursuit 의 r2 :  0.3810894583550424
# OrthogonalMatchingPursuitCV 의 r2 :  0.5582820621454118
# PLSCanonical 의 r2 :  -0.3905138164401407
# PLSRegression 의 r2 :  0.565451816246832
# PassiveAggressiveRegressor 의 r2 :  0.55768959691772
# PoissonRegressor 의 r2 :  0.5948970566851708
# RANSACRegressor 의 r2 :  0.448718344901593
# RadiusNeighborsRegressor 의 r2 :  0.25381479973940624
# RandomForestRegressor 의 r2 :  0.7754376695574688
# Ridge 의 r2 :  0.566401157995615
# RidgeCV 의 r2 :  0.5664011579956122
# SGDRegressor 의 r2 :  0.5609908583891307
# SVR 의 r2 :  0.4253395166480469
# TheilSenRegressor 의 r2 :  0.5565468541302145
# TransformedTargetRegressor 의 r2 :  0.567708352772792
# TweedieRegressor 의 r2 :  0.11487154813153178