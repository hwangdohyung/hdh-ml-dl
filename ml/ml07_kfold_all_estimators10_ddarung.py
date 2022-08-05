# 데이콘 따릉이 문제풀이 
import numpy as np 
import pandas as pd # csv 파일 당겨올 때 사용
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split,KFold,cross_val_score
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

# ARDRegression 의 r2 :  [0.51094341 0.59152147 0.60829279 0.58823227 0.60363157]
# AdaBoostRegressor 의 r2 :  [0.5021122  0.5587333  0.64666858 0.59523479 0.53852556]
# BaggingRegressor 의 r2 :  [0.71799402 0.73638966 0.78386264 0.74168722 0.7931741 ]
# BayesianRidge 의 r2 :  [0.50059412 0.57360587 0.60416837 0.59238667 0.60068489]
# CCA 의 r2 :  [-0.05472715  0.12092942  0.38457029  0.29610656  0.17217105]
# DecisionTreeRegressor 의 r2 :  [0.48602411 0.53626834 0.69853511 0.49514903 0.57591972]
# DummyRegressor 의 r2 :  [-1.06844486e-06 -8.22274615e-03 -3.89996785e-03 -8.86961127e-03
#  -5.56153108e-03]
# ElasticNet 의 r2 :  [0.49564855 0.56879039 0.59386249 0.58037631 0.59312537]
# ElasticNetCV 의 r2 :  [0.45672766 0.54319663 0.55794622 0.53032663 0.54448917]
# ExtraTreeRegressor 의 r2 :  [0.48290127 0.52755233 0.60493954 0.52033383 0.54007632]
# ExtraTreesRegressor 의 r2 :  [0.79310047 0.78573975 0.8288833  0.79981628 0.81214846]
# GammaRegressor 의 r2 :  [-8.03578326e-07 -5.32111746e-03 -2.54927974e-03 -6.52643224e-03
#  -3.73152756e-03]
# GaussianProcessRegressor 의 r2 :  [-2.00496801 -1.68541951 -1.70993776 -1.88644211 -1.65623099]
# GradientBoostingRegressor 의 r2 :  [0.74206624 0.73011221 0.79349179 0.77846501 0.78200958]
# HistGradientBoostingRegressor 의 r2 :  [0.74823854 0.77251848 0.82876427 0.79532989 0.81611511]
# HuberRegressor 의 r2 :  [0.48889596 0.56447486 0.57500082 0.56234555 0.58549432]
# IsotonicRegression 의 r2 :  [nan nan nan nan nan]
# KNeighborsRegressor 의 r2 :  [0.30625537 0.26292446 0.42952452 0.40449154 0.33462331]
# KernelRidge 의 r2 :  [0.50412715 0.58978318 0.60825663 0.59066981 0.59808727]
# Lars 의 r2 :  [0.50717332 0.58924277 0.60930529 0.59047738 0.60451426]
# LarsCV 의 r2 :  [0.50717332 0.58924277 0.59095966 0.58875565 0.60472822]
# Lasso 의 r2 :  [0.49838882 0.57429021 0.59946509 0.58751177 0.59824853]
# LassoCV 의 r2 :  [0.49882057 0.56587598 0.58288129 0.5656867  0.58765321]
# LassoLars 의 r2 :  [0.33620042 0.32393138 0.28736021 0.29995657 0.33061674]
# LassoLarsCV 의 r2 :  [0.50717332 0.58924277 0.6031874  0.58875565 0.60472822]
# LassoLarsIC 의 r2 :  [0.51273251 0.5914517  0.60726859 0.58888997 0.60417292]
# LinearRegression 의 r2 :  [0.50717332 0.58924277 0.60930529 0.59047738 0.60451426]
# LinearSVR 의 r2 :  [ 0.37881833 -0.39805028  0.07926091  0.3206138   0.5868559 ]
# MLPRegressor 의 r2 :  [0.50478539 0.56703862 0.60151043 0.56312572 0.60885982]
# MultiTaskElasticNet 의 r2 :  [nan nan nan nan nan]
# MultiTaskElasticNetCV 의 r2 :  [nan nan nan nan nan]
# MultiTaskLasso 의 r2 :  [nan nan nan nan nan]
# MultiTaskLassoCV 의 r2 :  [nan nan nan nan nan]
# NuSVR 의 r2 :  [-0.01298075  0.09699675  0.05638987  0.02758359  0.09029394]
# OrthogonalMatchingPursuit 의 r2 :  [0.28281675 0.31546766 0.369908   0.33529211 0.39700932]
# OrthogonalMatchingPursuitCV 의 r2 :  [0.49789496 0.579448   0.59113518 0.57432654 0.59866295]
# PLSCanonical 의 r2 :  [-0.92642071 -0.55170831 -0.09798047 -0.1505343  -0.60939897]
# PLSRegression 의 r2 :  [0.50037152 0.56957989 0.60371941 0.58488749 0.60480898]
# PassiveAggressiveRegressor 의 r2 :  [  0.07404996 -18.66844187   0.47627165   0.361828    -0.23501867]
# PoissonRegressor 의 r2 :  [-1.05868872e-06 -7.78097038e-03 -3.83673954e-03 -9.03220018e-03
#  -5.31975262e-03]
# RANSACRegressor 의 r2 :  [0.44814766 0.52602544 0.5188329  0.47707843 0.47930684]
# RadiusNeighborsRegressor 의 r2 :  [nan nan nan nan nan]
# RandomForestRegressor 의 r2 :  [0.76321911 0.75671868 0.80717775 0.77839524 0.78636126]
# Ridge 의 r2 :  [0.50312987 0.58829926 0.60738623 0.59304225 0.60084629]
# RidgeCV 의 r2 :  [0.5061134  0.58948157 0.60897326 0.591954   0.60335155]
# SGDRegressor 의 r2 :  [-3.81652064e+25 -2.03564253e+26 -1.77979737e+25 -1.06260862e+26
#  -6.74826640e+24]
# SVR 의 r2 :  [-0.00945393  0.09607805  0.0792113   0.04010016  0.09199237]
# TheilSenRegressor 의 r2 :  [0.50349508 0.56387951 0.5872635  0.56463691 0.59368371]
# TransformedTargetRegressor 의 r2 :  [0.50717332 0.58924277 0.60930529 0.59047738 0.60451426]
# TweedieRegressor 의 r2 :  [0.48300163 0.56193172 0.58208175 0.5581702  0.56331336]