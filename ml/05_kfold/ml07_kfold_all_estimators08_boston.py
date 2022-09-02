from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler


#1.데이터
datasets = load_boston()

x = datasets['data']
y = datasets['target']


# x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)


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

# ARDRegression 의 r2 :  [0.80125693 0.76317071 0.56809285 0.6400258  0.71991866]
# AdaBoostRegressor 의 r2 :  [0.91192233 0.78226746 0.78170815 0.82866872 0.89178355]
# BaggingRegressor 의 r2 :  [0.91331156 0.84760896 0.81824392 0.87852076 0.89650172]
# BayesianRidge 의 r2 :  [0.79379186 0.81123808 0.57943979 0.62721388 0.70719051]
# CCA 의 r2 :  [0.79134772 0.73828469 0.39419624 0.5795108  0.73224276]
# DecisionTreeRegressor 의 r2 :  [0.7972297  0.79190054 0.82851151 0.73882919 0.73092977]
# DummyRegressor 의 r2 :  [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911]
# ElasticNet 의 r2 :  [0.73383355 0.76745241 0.59979782 0.60616114 0.64658354]
# ElasticNetCV 의 r2 :  [0.71677604 0.75276545 0.59116613 0.59289916 0.62888608]
# ExtraTreeRegressor 의 r2 :  [0.77715923 0.6898418  0.67183089 0.69755076 0.83964474]
# ExtraTreesRegressor 의 r2 :  [0.93588633 0.8590596  0.78233136 0.88545967 0.92302272]
# GammaRegressor 의 r2 :  [-0.00058757 -0.03146716 -0.00463664 -0.02807276 -0.00298635]
# GaussianProcessRegressor 의 r2 :  [-6.07310526 -5.51957093 -6.33482574 -6.36383476 -5.35160828]
# GradientBoostingRegressor 의 r2 :  [0.94561442 0.83410717 0.82555452 0.88557887 0.93185404]
# HistGradientBoostingRegressor 의 r2 :  [0.93235978 0.82415907 0.78740524 0.88879806 0.85766226]
# HuberRegressor 의 r2 :  [0.70881407 0.65909542 0.53203819 0.36322935 0.62953938]
# IsotonicRegression 의 r2 :  [nan nan nan nan nan]
# KNeighborsRegressor 의 r2 :  [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856]
# KernelRidge 의 r2 :  [0.83333255 0.76712443 0.5304997  0.5836223  0.71226555]
# Lars 의 r2 :  [0.77467361 0.79839316 0.5903683  0.64083802 0.68439384]
# LarsCV 의 r2 :  [0.80141197 0.77573678 0.57807429 0.60068407 0.70833854]
# Lasso 의 r2 :  [0.7240751  0.76027388 0.60141929 0.60458689 0.63793473]
# LassoCV 의 r2 :  [0.71314939 0.79141061 0.60734295 0.61617714 0.66137127]
# LassoLars 의 r2 :  [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911]
# LassoLarsCV 의 r2 :  [0.80301044 0.77573678 0.57807429 0.60068407 0.72486787]
# LassoLarsIC 의 r2 :  [0.81314239 0.79765276 0.59012698 0.63974189 0.72415009]
# LinearRegression 의 r2 :  [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215]
# LinearSVR 의 r2 :  [0.57470919 0.72471781 0.51731868 0.41662826 0.65193847]
# MLPRegressor 의 r2 :  [0.62044701 0.60574663 0.48843393 0.39372355 0.55901924]
# MultiTaskElasticNet 의 r2 :  [nan nan nan nan nan]
# MultiTaskElasticNetCV 의 r2 :  [nan nan nan nan nan]
# MultiTaskLasso 의 r2 :  [nan nan nan nan nan]
# MultiTaskLassoCV 의 r2 :  [nan nan nan nan nan]
# NuSVR 의 r2 :  [0.2594254  0.33427351 0.263857   0.11914968 0.170599  ]
# OrthogonalMatchingPursuit 의 r2 :  [0.58276176 0.565867   0.48689774 0.51545117 0.52049576]
# OrthogonalMatchingPursuitCV 의 r2 :  [0.75264599 0.75091171 0.52333619 0.59442374 0.66783377]
# PLSCanonical 의 r2 :  [-2.23170797 -2.33245351 -2.89155602 -2.14746527 -1.44488868]
# PLSRegression 의 r2 :  [0.80273131 0.76619347 0.52249555 0.59721829 0.73503313]
# PassiveAggressiveRegressor 의 r2 :  [-0.88039522  0.18236354 -0.07746659 -0.28020812 -0.34497781]
# PoissonRegressor 의 r2 :  [0.85570647 0.81899779 0.66801489 0.67994598 0.7670857 ]
# RANSACRegressor 의 r2 :  [0.72654757 0.69426221 0.49720918 0.36914063 0.17548419]
# RadiusNeighborsRegressor 의 r2 :  [nan nan nan nan nan]
# RandomForestRegressor 의 r2 :  [0.92021884 0.85268775 0.81764982 0.89040366 0.91117783]
# Ridge 의 r2 :  [0.80984876 0.80618063 0.58111378 0.63459427 0.72264776]
# RidgeCV 의 r2 :  [0.81125292 0.80010535 0.58888303 0.64008984 0.72362912]
# SGDRegressor 의 r2 :  [-7.98747459e+25 -9.23335067e+25 -1.03026194e+27 -1.65075094e+26
#  -6.27372772e+24]
# SVR 의 r2 :  [0.23475113 0.31583258 0.24121157 0.04946335 0.14020554]
# TheilSenRegressor 의 r2 :  [0.77586084 0.7198516  0.58428856 0.54710154 0.72023358]
# TransformedTargetRegressor 의 r2 :  [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215]
# TweedieRegressor 의 r2 :  [0.7320775  0.75549621 0.57408841 0.57661534 0.63094693]