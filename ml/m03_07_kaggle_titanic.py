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

                                          
#2.모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron,LogisticRegression  #logisticregression : regression 이 들어가지만 분류다!
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model1 = SVC()
model2 = Perceptron()
model3 = LogisticRegression()
model4 = KNeighborsClassifier()
model5 = DecisionTreeClassifier()
model6 = RandomForestClassifier()

#3.컴파일,훈련

model1.fit(x_train,y_train)#callback 리스트형태 더 호출할수있다.
model2.fit(x_train,y_train)#callback 리스트형태 더 호출할수있다.
model3.fit(x_train,y_train)#callback 리스트형태 더 호출할수있다.
model4.fit(x_train,y_train)#callback 리스트형태 더 호출할수있다.
model5.fit(x_train,y_train)#callback 리스트형태 더 호출할수있다.
model6.fit(x_train,y_train)#callback 리스트형태 더 호출할수있다.

#4.평가,예측

result1 = model1.score(x_test, y_test)
print('SCV : ', result1)


result2 = model2.score(x_test, y_test)
print('Perceptron : ', result2)

result3 = model3.score(x_test, y_test)
print('LogisticRegression : ', result3)


result4 = model4.score(x_test, y_test)
print('KNeighborsClassifier : ', result4)


result5 = model5.score(x_test, y_test)
print('DecisionTreeClassifier : ', result5)

result6 = model6.score(x_test, y_test)
print('RandomForestClassifier : ', result6)

# SCV :  0.8134328358208955
# Perceptron :  0.753731343283582
# LogisticRegression :  0.7611940298507462
# KNeighborsClassifier :  0.8171641791044776
# DecisionTreeClassifier :  0.7873134328358209
# RandomForestClassifier :  0.8097014925373134