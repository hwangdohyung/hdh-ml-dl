from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler


#1.데이터
datasets = fetch_california_housing()

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

# ARDRegression 의 r2 :  [0.61508972 0.58238965 0.58600404 0.58646746 0.5952875 ]
# AdaBoostRegressor 의 r2 :  [0.42743726 0.4487193  0.45403184 0.38670803 0.38637489]
# BaggingRegressor 의 r2 :  [0.80668049 0.79726942 0.77262426 0.77889712 0.79023456]
# BayesianRidge 의 r2 :  [0.61611062 0.59276859 0.59481212 0.59811394 0.60723539]
# CCA 의 r2 :  [0.5787713  0.55665247 0.55316522 0.55422957 0.56913271]
# DecisionTreeRegressor 의 r2 :  [0.62141103 0.59995843 0.60748715 0.5989069  0.60747814]
# DummyRegressor 의 r2 :  [-3.24186381e-04 -5.02433932e-05 -7.66495474e-06 -8.44220662e-05
#  -1.44274084e-03]
# ElasticNet 의 r2 :  [0.43670132 0.42774727 0.40947393 0.4125726  0.42624816]
# ElasticNetCV 의 r2 :  [0.59614738 0.59325843 0.47681049 0.56841551 0.58887966]
# ExtraTreeRegressor 의 r2 :  [0.58338674 0.58429801 0.54481628 0.52024707 0.52210754]
# ExtraTreesRegressor 의 r2 :  [0.82582612 0.81874139 0.80110327 0.80467061 0.81323992]
# GammaRegressor 의 r2 :  [-3.30948917e-04 -5.05283764e-05 -7.74419254e-06 -8.39766694e-05
#  -1.42877923e-03]
# GaussianProcessRegressor 의 r2 :  [-2.72619448 -2.83519402 -2.80513819 -2.83791675 -2.78069256]
# GradientBoostingRegressor 의 r2 :  [0.80054583 0.78656725 0.77711305 0.78188733 0.79162359]
# HistGradientBoostingRegressor 의 r2 :  [0.84821902 0.83378115 0.82335652 0.8290065  0.84021775]
# HuberRegressor 의 r2 :  [-11.48774679   0.52577644   0.06050617   0.49706096   0.45274533]
# IsotonicRegression 의 r2 :  [nan nan nan nan nan]
# KNeighborsRegressor 의 r2 :  [0.15850596 0.14291154 0.15123397 0.1535585  0.16412787]
# KernelRidge 의 r2 :  [0.54605307 0.53734866 0.53483471 0.53621985 0.54127574]
# Lars 의 r2 :  [0.61614066 0.59250746 0.59486463 0.5981755  0.60724957]
# LarsCV 의 r2 :  [0.61666036 0.59250746 0.42823193 0.5981755  0.60682002]
# Lasso 의 r2 :  [0.28647943 0.28780898 0.27928678 0.2801791  0.28893083]
# LassoCV 의 r2 :  [0.60073264 0.59644162 0.4847404  0.57376844 0.59131269]
# LassoLars 의 r2 :  [-3.24186381e-04 -5.02433932e-05 -7.66495474e-06 -8.44220662e-05
#  -1.44274084e-03]
# LassoLarsCV 의 r2 :  [0.61666036 0.59250746 0.42823193 0.5981755  0.60682002]
# LassoLarsIC 의 r2 :  [0.61614066 0.59347925 0.59471286 0.5981755  0.60724957]
# LinearRegression 의 r2 :  [0.61614066 0.59250746 0.59486463 0.5981755  0.60724957]
# LinearSVR 의 r2 :  [-5.88302264 -3.82106794 -2.67930953  0.11865596  0.29931834]
# MLPRegressor 의 r2 :  [0.53959477 0.18433108 0.3090466  0.50580753 0.06354239]
# MultiTaskElasticNet 의 r2 :  [nan nan nan nan nan]
# MultiTaskElasticNetCV 의 r2 :  [nan nan nan nan nan]
# MultiTaskLasso 의 r2 :  [nan nan nan nan nan]
# MultiTaskLassoCV 의 r2 :  [nan nan nan nan nan]
# NuSVR 의 r2 :  [0.00497904 0.00590294 0.00689076 0.00567127 0.01731536]
# OrthogonalMatchingPursuit 의 r2 :  [0.49497108 0.47987668 0.45729284 0.45909253 0.47373082]
# OrthogonalMatchingPursuitCV 의 r2 :  [0.61783292 0.59933302 0.49332567 0.58212067 0.59210345]
# PLSCanonical 의 r2 :  [0.35849242 0.37071128 0.37797438 0.38564743 0.36020325]
# PLSRegression 의 r2 :  [0.52503134 0.52940822 0.51350893 0.51448473 0.52158067]
# PassiveAggressiveRegressor 의 r2 :  [-2.556689   -1.2478947  -0.10721954 -1.13788717 -0.66037452]
# PoissonRegressor 의 r2 :  [-3.47517228e-04 -5.34833349e-05 -8.16959358e-06 -8.94134809e-05
#  -1.52431824e-03]
# RANSACRegressor 의 r2 :  [ 0.17560334 -0.12112539 -6.9179908   0.41345286  0.4487429 ]
# RadiusNeighborsRegressor 의 r2 :  [nan nan nan nan nan]
# RandomForestRegressor 의 r2 :  [0.82534193 0.81666207 0.79685348 0.80235786 0.8078687 ]
# Ridge 의 r2 :  [0.61613387 0.59258383 0.59485303 0.59816207 0.60724643]
# RidgeCV 의 r2 :  [0.61607032 0.59324453 0.59474791 0.59804082 0.60721579]
# SGDRegressor 의 r2 :  [-2.31861562e+27 -4.04440657e+30 -4.49968991e+28 -2.38642964e+28
#  -1.52933676e+29]
# SVR 의 r2 :  [-0.02656953 -0.02789739 -0.02674677 -0.02931755 -0.00772644]
# TheilSenRegressor 의 r2 :  [-40.68386975   0.20713314 -10.26786968   0.59131407   0.46761839]
# TransformedTargetRegressor 의 r2 :  [0.61614066 0.59250746 0.59486463 0.5981755  0.60724957]
# TweedieRegressor 의 r2 :  [0.49136427 0.49952145 0.4780168  0.48018093 0.49776091]