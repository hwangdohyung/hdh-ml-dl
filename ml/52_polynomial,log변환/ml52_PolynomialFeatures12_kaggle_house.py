import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.experimental import enable_halving_search_cv # 실험적 버전 정식버전이 아니다 *
from sklearn.model_selection import train_test_split,KFold,GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV,StratifiedKFold
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

from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LinearRegression
 
x,y = train_set,trainLabel

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True,random_state=123)

kfold = KFold(n_splits=5, shuffle=True, random_state=123)
 
 
 
model = make_pipeline(StandardScaler(),LinearRegression(),)
model.fit(x_train,y_train)
print('그냥: ' , model.score(x_test,y_test)) 

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring= 'r2')
print('그냥CV : ', scores)
print('그냥CV 엔빵 : ', np.mean(scores))


##################### polynomial 후 ########################

pf = PolynomialFeatures(degree=2, 
                         include_bias=False
                        )

xp = pf.fit_transform(x)
print(xp.shape)             

x_train,x_test,y_train,y_test = train_test_split(xp,y, train_size=0.8, shuffle=True,random_state=123)

#2.모델 
model = make_pipeline(StandardScaler(),LinearRegression(),)
model.fit(x_train,y_train)
print('poly : ' , model.score(x_test,y_test)) 

scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring= 'r2')
print('polyCV : ', scores)
print('polyCV 엔빵 : ', np.mean(scores))

# 그냥:  0.8689230163645146
# 그냥CV :  [ 3.52489711e-01  8.42650466e-01  8.59260150e-01  8.09170837e-01
#  -3.11342307e+21]
# 그냥CV 엔빵 :  -6.226846132992509e+20
# (1460, 2849)
# poly :  0.317179910647995
# polyCV :  [-4.06102362  0.39649741 -0.44818195  0.43885025  0.60156841]
# polyCV 엔빵 :  -0.6144579013973772