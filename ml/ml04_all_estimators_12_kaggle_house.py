import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from sklearn.impute import SimpleImputer

#1.data 처리
############# id컬럼 index 처리 ##########
path = './_data/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')
train_set.set_index('Id', inplace=True)
test_set.set_index('Id', inplace=True)
test_id_index = train_set.index
trainLabel = train_set['SalePrice']
train_set.drop(['SalePrice'], axis=1, inplace=True)
############################################
################### 트레인,테스트 합치기 ##################
alldata = pd.concat((train_set, test_set), axis=0)
alldata_index = alldata.index
################## NA 값 20프로 이상은 drop! ##########
NA_Ratio = 0.8 * len(alldata)
alldata.dropna(axis=1, thresh=NA_Ratio, inplace=True)

################### 수치형,카테고리형 분리,범위 설정 #############
alldata_obj = alldata.select_dtypes(include='object') 
alldata_num = alldata.select_dtypes(exclude='object')

for objList in alldata_obj:
    alldata_obj[objList] = LabelEncoder().fit_transform(alldata_obj[objList].astype(str))
##################### 소수 na 값 처리 ###################    
imputer = SimpleImputer(strategy='mean')
imputer.fit(alldata_num)
alldata_impute = imputer.transform(alldata_num)
alldata_num = pd.DataFrame(alldata_impute, columns=alldata_num.columns, index=alldata_index)  
###################### 분리한 데이터 다시 합치기 #####################
alldata = pd.merge(alldata_obj, alldata_num, left_index=True, right_index=True)  
##################### 트레인, 테스트 다시 나누기 ##################
train_set = alldata[:len(train_set)]
test_set = alldata[len(train_set):]
############### 트레인 데이터에 sale price 합치기 ##############
train_set['SalePrice'] = trainLabel
############### sale price 다시 드랍 #####################
train_set = train_set.drop(['SalePrice'], axis =1)
print(train_set)
print(trainLabel)
print(test_set)


###############################################################
x_train, x_test, y_train, y_test = train_test_split(train_set, trainLabel, train_size=0.8, 
                                            
                                                random_state=58)


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
        
        
# ARDRegression 의 r2 :  0.8210477064668218
# AdaBoostRegressor 의 r2 :  0.806216077845067
# BaggingRegressor 의 r2 :  0.8177297753412818
# BayesianRidge 의 r2 :  -0.0034535069645778016
# CCA 의 r2 :  -0.3656169274214629
# DecisionTreeRegressor 의 r2 :  0.8100323501062745
# DummyRegressor 의 r2 :  -0.003455010869476638
# ElasticNet 의 r2 :  0.7961725520649962
# ElasticNetCV 의 r2 :  0.06526063198418841
# ExtraTreeRegressor 의 r2 :  0.7052021072514396
# ExtraTreesRegressor 의 r2 :  0.8817550033444084
# GammaRegressor 의 r2 :  -0.00345501086947686
# GaussianProcessRegressor 의 r2 :  -4.392249084892602
# GradientBoostingRegressor 의 r2 :  0.8932151404702329
# HistGradientBoostingRegressor 의 r2 :  0.8692525441315931
# HuberRegressor 의 r2 :  0.2656569093022262
# KNeighborsRegressor 의 r2 :  0.4863318099651168
# KernelRidge 의 r2 :  0.8169001171226397
# LarsCV 의 r2 :  0.8023207211280257
# Lasso 의 r2 :  0.8193201517611958
# LassoCV 의 r2 :  0.8201866257473551
# LassoLars 의 r2 :  0.819900963477243
# LassoLarsCV 의 r2 :  0.825865657446534
# LassoLarsIC 의 r2 :  0.8252721451158544
# LinearRegression 의 r2 :  0.8193063662931154
# LinearSVR 의 r2 :  -3.1908534141810527
# MLPRegressor 의 r2 :  -3.606401376901788
# NuSVR 의 r2 :  -0.002802369152764106
# OrthogonalMatchingPursuit 의 r2 :  0.8321531170427163
# OrthogonalMatchingPursuitCV 의 r2 :  0.8321531170427163
# PLSCanonical 의 r2 :  -4.0602758000712305
# PLSRegression 의 r2 :  0.8345101248461182
# PassiveAggressiveRegressor 의 r2 :  0.15927040146012728
# PoissonRegressor 의 r2 :  -0.00345501086947686
# RANSACRegressor 의 r2 :  0.8060391881619706
# RadiusNeighborsRegressor 의 r2 :  -1.2538463919904247e+28
# RandomForestRegressor 의 r2 :  0.8841664908635444
# Ridge 의 r2 :  0.819275274838131
# RidgeCV 의 r2 :  0.8185663900013733
# SGDRegressor 의 r2 :  -1.3500152915289162e+21
# SVR 의 r2 :  -0.029139138199599346