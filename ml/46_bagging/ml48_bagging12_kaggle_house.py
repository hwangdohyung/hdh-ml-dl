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


###############################################################
x_train, x_test, y_train, y_test = train_test_split(train_set, trainLabel, train_size=0.9, 
                                            
                                                random_state=58)
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle= True, random_state=66)

parameters = {'RF__n_estimators' : [100],
              'RF__learning_rate': [0.3],
              'RF__max_depth': [3],
              'RF__gamma': [1],
              'RF__min_child_weight': [0.001],
              'RF__subsample': [0.1],
              'RF__colsample_bytree': [0.5],
              'RF__colsample_bylevel': [0.7],
              'RF__colsample_bynode': [0.2] ,
              'RF__reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10],
              'RF__reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10]
              }

#2. 모델 
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline 
from xgboost import XGBRegressor


pipe = Pipeline([('minmax', MinMaxScaler()), ('RF', XGBRegressor())],verbose=1) # '' 는 그냥 변수명 (파이프라인에 그리드서치 엮을때 모델명 변수를 파라미터에 명시해 줘야 됨.)


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

# model =RandomForestRegressor (pipe, parameters, cv=5, verbose=1)
model =RandomizedSearchCV(XGBRegressor(), parameters, cv=5, verbose=1)

model.fit(x_train, y_train) # 파이프라인에서 fit 할땐 스케일링의 transform 과 fit이 돌아간다. 

#4.평가, 예측 
result = model.score(x_test, y_test)

print('최상의 매개변수 : ', model.best_params_)
print('최상의 점수 : ', model.best_score_)


print('model.score : ', round(result,2))

# model.score :  0.89

# 최상의 점수 :  0.8838098026347219
# model.score :  0.91
