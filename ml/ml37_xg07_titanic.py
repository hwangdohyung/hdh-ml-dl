#[실습]#
from re import I
import numpy as np 
import pandas as pd
from sklearn import metrics
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Dropout,Input,Flatten,Conv2D
from sklearn.experimental import enable_halving_search_cv # 실험적 버전 정식버전이 아니다 *
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split,GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import tensorflow as tf
import datetime
from sklearn.model_selection import cross_val_score,cross_val_predict,KFold,StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1.데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv',index_col =0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

##########전처리############
train_test_data = [train_set, test_set]
sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

print(dataset)

for dataset in train_test_data:
    # 가족수 = 형제자매 + 부모님 + 자녀 + 본인
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    
    # 가족수 > 1이면 동승자 있음
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
embarked_mapping = {'S':0, 'C':1, 'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([\w]+)\.', expand=False)
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].apply(lambda x: 0 if x=="Mr" else 1 if x=="Miss" else 2 if x=="Mrs" else 3 if x=="Master" else 4)

train_set['Cabin'] = train_set['Cabin'].str[:1]
for dataset in train_test_data:
    dataset['Age'].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset['Agebin'] = pd.cut(dataset['Age'], 5, labels=[0,1,2,3,4])
for dataset in train_test_data:
    dataset["Fare"].fillna(dataset.groupby("Pclass")["Fare"].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset['Farebin'] = pd.qcut(dataset['Fare'], 4, labels=[0,1,2,3])
    drop_column = ['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']

for dataset in train_test_data:
    dataset = dataset.drop(drop_column, axis=1, inplace=True)
print(train_set.head())


x = train_set.drop(['Survived'], axis=1,)
y = train_set['Survived']



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size= 0.3,random_state=61)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle= True, random_state=66)

parameters = {'RF__n_estimators' : [100, 200, 300, 400, 500, 1000],
            #   'RF__learning_rate': [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001],
            #   'RF__max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            #   'RF__gamma': [0, 1, 2, 3, 4, 5, 7, 10, 100],
            #   'RF__min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10],
            #   'RF__subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
            #   'RF__colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
            #   'RF__colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
            #   'RF__colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] ,
            #   'RF__reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10],
            #   'RF__reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10]
              } 

#2.모델구성
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline,Pipeline
from xgboost import XGBClassifier

# model = make_pipeline(MinMaxScaler(),RandomForestClassifier()) 
pipe = Pipeline([('minmax', MinMaxScaler()),('RF', XGBClassifier())],verbose=1)

#3.훈련     
model = GridSearchCV(pipe,parameters,cv=5,verbose=1)
model.fit(x_train,y_train)

#4.평가,예측
result = model.score(x_test, y_test)

print('model.score : ', round(result, 2))


# model.score :  0.79

