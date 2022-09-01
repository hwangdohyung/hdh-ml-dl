# m36_dacon_travel.py
import numpy as np
import pandas as pd                               
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score
from tqdm import tqdm_notebook
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

#1. 데이터
path = 'D:\study_data\_data\dacon_travle/'
train = pd.read_csv(path + 'train.csv',                 
                        index_col=0)                       

test = pd.read_csv(path + 'test.csv',                                   
                       index_col=0)

sample_submission = pd.read_csv(path + 'sample_submission.csv')

import random
import os
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(2022) # Seed 고정

print(train.describe()) 
print(test.describe()) 
print(train.shape)
print(test.shape)

print(train.isnull().sum())
# Age                          94
# TypeofContact                10
# CityTier                      0
# DurationOfPitch             102
# Occupation                    0
# Gender                        0
# NumberOfPersonVisiting        0
# NumberOfFollowups            13
# ProductPitched                0
# PreferredPropertyStar        10
# MaritalStatus                 0
# NumberOfTrips                57
# Passport                      0
# PitchSatisfactionScore        0
# OwnCar                        0
# NumberOfChildrenVisiting     27
# Designation                   0
# MonthlyIncome               100
# ProdTaken                     0
# median = data.median()
# print("평균:",median)
# data3 =data.fillna(median)
# print(data3)


# train['NumberOfFollowups'] = train['NumberOfFollowups'].fillna(train.groupby('Designation')['NumberOfFollowups'].transform('mean'), inplace=True)
# test['NumberOfFollowups'] = test['NumberOfFollowups'].fillna(test.groupby('Designation')['NumberOfFollowups'].transform('mean'), inplace=True)

train['NumberOfFollowups'].fillna(train.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('median'), inplace=True)
test['NumberOfFollowups'].fillna(test.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('median'), inplace=True)

train.loc[ train['Gender'] =='Fe Male' , 'Gender'] = 'Female'
test.loc[ test['Gender'] =='Fe Male' , 'Gender'] = 'Female'

train['Age'].fillna(train.groupby('Designation')['Age'].transform('mean'), inplace=True)
test['Age'].fillna(test.groupby('Designation')['Age'].transform('mean'), inplace=True)

train['TypeofContact'].fillna('Self Enquiry', inplace=True)
test['TypeofContact'].fillna('Self Enquiry', inplace=True)

train['MonthlyIncome'].fillna(train.groupby('Designation')['MonthlyIncome'].transform('median'), inplace=True)
test['MonthlyIncome'].fillna(test.groupby('Designation')['MonthlyIncome'].transform('median'), inplace=True)

# train['DurationOfPitch']=train['DurationOfPitch'].fillna(train['DurationOfPitch'].median())
# test['DurationOfPitch']=test['DurationOfPitch'].fillna(test['DurationOfPitch'].median())

train['DurationOfPitch']=train['DurationOfPitch'].fillna(0)
test['DurationOfPitch']=test['DurationOfPitch'].fillna(0)

# train['PreferredPropertyStar'].fillna(train.groupby('Occupation')['PreferredPropertyStar'].transform('median'), inplace=True)
# test['PreferredPropertyStar'].fillna(test.groupby('Occupation')['PreferredPropertyStar'].transform('median'), inplace=True)

train['PreferredPropertyStar'].fillna(0)
test['PreferredPropertyStar'].fillna(0)

# combine = [train,test]
# for dataset in combine:    
#     dataset.loc[ dataset['Age'] <= 26.6, 'Age'] = 0
#     dataset.loc[(dataset['Age'] > 26.6) & (dataset['Age'] <= 35.2), 'Age'] = 1
#     dataset.loc[(dataset['Age'] > 35.2) & (dataset['Age'] <= 43.8), 'Age'] = 2
#     dataset.loc[(dataset['Age'] > 43.8) & (dataset['Age'] <= 52.4), 'Age'] = 3
#     dataset.loc[ dataset['Age'] > 52.4, 'Age'] = 4


print(train.isnull().sum())

# 탐색경로', '후속조치수', '프리젠테이션기간', '선호숙박등급', '연간여행횟수', '미취학아동' median()
# 'TypeofContact', 'NumberOfFollowups','DurationOfPitch', 'PreferredPropertyStar','NumberOfChildrenVisiting','NumberOfTrips'  

print(train.info())
print(test.info())


# 결측치를 처리하는 함수를 작성.
def handle_na(data):
    temp = data.copy()
    for col, dtype in temp.dtypes.items():
        if dtype == 'object':
            # 문자형 칼럼의 경우 'Unknown'
            value = 'Unknown'
        elif dtype == int or dtype == float:
            # 수치형 칼럼의 경우 0
            value = 0
        temp.loc[:,col] = temp[col].fillna(value)
    return temp

train_nona = handle_na(train)

# 결측치 처리가 잘 되었는지 확인
train_nona.isna().sum()

print(train_nona.isna().sum())
object_columns = train_nona.columns[train_nona.dtypes == 'object']
print('object 칼럼 : ', list(object_columns))

train_nona[object_columns]

print(train_nona.shape)
print(test.shape)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(train_nona['TypeofContact'])

#학습된 encoder를 사용하여 문자형 변수를 숫자로 변환
encoder.transform(train_nona['TypeofContact'])
print(train_nona['TypeofContact'])

train_enc = train_nona.copy()

# 모든 문자형 변수에 대해 encoder를 적용
for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(train_enc[o_col])
    train_enc[o_col] = encoder.transform(train_enc[o_col])

# 결과를 확인
print(train_enc)
# 결측치 처리
test = handle_na(test)

# 문자형 변수 전처리
for o_col in object_columns:
    encoder = LabelEncoder()
    
    # test 데이터를 이용해 encoder를 학습하는 것은 Data Leakage 입니다! 조심!
    encoder.fit(train_nona[o_col])
    
    # test 데이터는 오로지 transform 에서만 사용되어야 합니다.
    test[o_col] = encoder.transform(test[o_col])

# 결과를 확인
print(test)



print(train_enc.describe())  # DurationOfPitch, MonthlyIncome
print("=============================상관계수 히트 맵==============")
print(train_enc.corr())                    # 상관관계를 확인.  
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(font_scale=0.7)
sns.heatmap(data=train_enc.corr(),square=True, annot=True, cbar=True) 
# plt.show()


# 모델 선언
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from xgboost import XGBClassifier, XGBRegressor 
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# import matplotlib.pyplot as plt

# train_enc.plot.box()
# plt.title('boston')
# plt.xlabel('Feature')
# plt.ylabel('data')
# plt.show()

# 분석할 의미가 없는 칼럼을 제거합니다.
# 상관계수 그래프를 통해 연관성이 적은것과 - 인것을 빼준다.
train = train_enc.drop(columns=['NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar', 'MonthlyIncome', 'NumberOfTrips','NumberOfFollowups','Designation'])  
test = test.drop(columns=['NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar', 'MonthlyIncome', 'NumberOfTrips','NumberOfFollowups','Designation'])
# 'TypeofContact','NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar', 'MonthlyIncoe'

# 탐색경로', '후속조치수', '프리젠테이션기간', '선호숙박등급', '연간여행횟수', '미취학아동' median()
# 'TypeofContact', 'NumberOfFollowups','DurationOfPitch', 'PreferredPropertyStar','NumberOfChildrenVisiting','NumberOfTrips'  

# 학습에 사용할 정보와 예측하고자 하는 정보를 분리합니다.


x = train.drop(columns=['ProdTaken'])
y = train[['ProdTaken']]

print(x.isnull().sum())

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer,SimpleImputer
imp = IterativeImputer(estimator = LinearRegression(), 
                       tol= 1e-10, 
                       max_iter=30, 
                       verbose=2, 
                       imputation_order='roman')
x = pd.DataFrame(imp.fit_transform(x))

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state=42, train_size=0.87,shuffle=True)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split, KFold , StratifiedKFold
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# 모델 학습
# xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, gamma = 1, subsample=1, colsample_bytree = 1, max_depth=4,random_state=123)


# ##########################GridSearchCV###############################
# n_splits = 5

# parameters = {'n_estimators':[1000],
#               'learning_rate':[0.1],
#               'max_depth':[3],
#               'gamma': [0],
#             #   'min_child_weight':[1],
#               'subsample':[1],
#               'colsample_bytree':[1],
#             #   'colsample_bylevel':[1],
#             #   'colsample_byload':[1],
#             #   'reg_alpha':[0],
#             #   'reg_lambda':[1]
#               }  

# kfold = KFold(n_splits=n_splits ,shuffle=True, random_state=123)
# xgb = XGBClassifier(random_state=123,
#                     )
# model = GridSearchCV(xgb,param_grid=parameters, cv =kfold, n_jobs=8)
##########################GridSearchCV###############################

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
############################0821_1####################################
param_grid = [
              {'n_estimators':[10], 'max_features':[10]},
              {'bootstrap':[False],'n_estimators':[400], 'max_features':[6]}
]

forest_reg =  RandomForestClassifier()
# 
model = RandomizedSearchCV(forest_reg, param_grid, cv=5,
                           scoring='accuracy',
                           verbose=0,
                           return_train_score=True)

############################0821_1####################################

# model = RandomForestClassifier()

# model = ExtraTreesClassifier(n_estimators=100, random_state=2022)

model.fit(x_train,y_train)

prediction = model.predict(x_test)
prediction1 = model.predict(test)

print('----------------------예측된 데이터의 상위 10개의 값 확인--------------------\n')

print('acc : ', accuracy_score(prediction,y_test))

print(prediction[0:11])
# print(model.score(x_train, y_train))
# 예측된 값을 정답파일과 병합
print(prediction.shape)

sample_submission['ProdTaken'] = prediction1

# 정답파일 데이터프레임 확인
print(sample_submission[:15])

sample_submission.to_csv(path+'sample_submission0831_2.csv',index = False)

