#[실습]#
import numpy as np 
import pandas as pd
from sklearn import metrics
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Dropout,Input,Flatten,Conv2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import tensorflow as tf
import datetime

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

print(x_train.shape,x_test.shape)

# minmax , standard
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)        

                                          
from sklearn.utils import all_estimators
#2.모델구성
allAlgorithms = all_estimators(type_filter='classifier')  # 분류모델 전부를 보여준다 
# allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgorithms : ', allAlgorithms)
print('모델의 갯수 : ', len(allAlgorithms))
import warnings
warnings.filterwarnings('ignore') # 출력만 안해준다

for (name, algorithm) in allAlgorithms:       # 리스트 안에 키밸류(알고리즘 이름과,위치)를 받아서 반복한다.
    try:                                      # 이것을 진행해 
        model = algorithm()
        model.fit(x_train,y_train)
    
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test,y_predict)
        print(name, '의 정답율 : ', acc)              
    except:                                   # 에러가 뜨면 계속 진행해
        continue
        # print(name, '안나온놈')
        
# AdaBoostClassifier 의 정답율 :  0.7723880597014925
# BaggingClassifier 의 정답율 :  0.7873134328358209
# BernoulliNB 의 정답율 :  0.7761194029850746
# CalibratedClassifierCV 의 정답율 :  0.7985074626865671
# CategoricalNB 의 정답율 :  0.7313432835820896
# ComplementNB 의 정답율 :  0.7388059701492538
# DecisionTreeClassifier 의 정답율 :  0.7985074626865671
# DummyClassifier 의 정답율 :  0.6231343283582089
# ExtraTreeClassifier 의 정답율 :  0.7947761194029851
# ExtraTreesClassifier 의 정답율 :  0.7985074626865671
# GaussianNB 의 정답율 :  0.7723880597014925
# GaussianProcessClassifier 의 정답율 :  0.7985074626865671
# GradientBoostingClassifier 의 정답율 :  0.8097014925373134
# HistGradientBoostingClassifier 의 정답율 :  0.8208955223880597
# KNeighborsClassifier 의 정답율 :  0.8171641791044776
# LabelPropagation 의 정답율 :  0.8134328358208955
# LabelSpreading 의 정답율 :  0.8097014925373134
# LinearDiscriminantAnalysis 의 정답율 :  0.7873134328358209
# LinearSVC 의 정답율 :  0.7798507462686567
# LogisticRegression 의 정답율 :  0.7611940298507462
# LogisticRegressionCV 의 정답율 :  0.7761194029850746
# MLPClassifier 의 정답율 :  0.7835820895522388
# MultinomialNB 의 정답율 :  0.75
# NearestCentroid 의 정답율 :  0.7425373134328358
# NuSVC 의 정답율 :  0.7947761194029851
# PassiveAggressiveClassifier 의 정답율 :  0.75
# Perceptron 의 정답율 :  0.753731343283582
# QuadraticDiscriminantAnalysis 의 정답율 :  0.753731343283582
# RadiusNeighborsClassifier 의 정답율 :  0.7686567164179104
# RandomForestClassifier 의 정답율 :  0.8022388059701493
# RidgeClassifier 의 정답율 :  0.7873134328358209
# RidgeClassifierCV 의 정답율 :  0.7873134328358209
# SGDClassifier 의 정답율 :  0.7574626865671642
# SVC 의 정답율 :  0.8134328358208955