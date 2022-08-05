#[실습]#
from re import I
import numpy as np 
import pandas as pd
from sklearn import metrics
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Dropout,Input,Flatten,Conv2D
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import tensorflow as tf
import datetime
from sklearn.model_selection import cross_val_score,cross_val_predict,KFold

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

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle= True, random_state=66)                                          


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
        
        scores = cross_val_score(model, x_train, y_train, cv= kfold)
        print(name ,'의 정답률 : ', scores, )
    
        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = accuracy_score(y_test,y_predict)
        print(name,'의 cross_val_predict ACC : ', acc)
        
    except:                                   # 에러가 뜨면 계속 진행해
        continue
        # print(name, '안나온놈')


# AdaBoostClassifier 의 정답률 :  [0.768      0.84       0.808      0.80645161 0.83870968]
# AdaBoostClassifier 의 cross_val_predict ACC :  0.7761194029850746
# BaggingClassifier 의 정답률 :  [0.752      0.856      0.8        0.80645161 0.80645161]
# BaggingClassifier 의 cross_val_predict ACC :  0.7910447761194029
# BernoulliNB 의 정답률 :  [0.712      0.824      0.824      0.75806452 0.83064516]
# BernoulliNB 의 cross_val_predict ACC :  0.7649253731343284
# CalibratedClassifierCV 의 정답률 :  [0.752      0.856      0.8        0.83064516 0.83870968]
# CalibratedClassifierCV 의 cross_val_predict ACC :  0.8022388059701493
# CategoricalNB 의 정답률 :  [0.672      0.792      0.736      0.76612903 0.84677419]
# CategoricalNB 의 cross_val_predict ACC :  0.7574626865671642
# ComplementNB 의 정답률 :  [0.728      0.816      0.8        0.75       0.82258065]
# ComplementNB 의 cross_val_predict ACC :  0.746268656716418
# DecisionTreeClassifier 의 정답률 :  [0.776     0.872     0.784     0.7983871 0.7983871]
# DecisionTreeClassifier 의 cross_val_predict ACC :  0.7835820895522388
# DummyClassifier 의 정답률 :  [0.592      0.624      0.632      0.59677419 0.62096774]
# DummyClassifier 의 cross_val_predict ACC :  0.6231343283582089
# ExtraTreeClassifier 의 정답률 :  [0.792      0.864      0.808      0.7983871  0.80645161]
# ExtraTreeClassifier 의 cross_val_predict ACC :  0.75
# ExtraTreesClassifier 의 정답률 :  [0.784      0.856      0.808      0.81451613 0.82258065]
# ExtraTreesClassifier 의 cross_val_predict ACC :  0.7761194029850746
# GaussianNB 의 정답률 :  [0.728      0.856      0.776      0.79032258 0.83064516]
# GaussianNB 의 cross_val_predict ACC :  0.7611940298507462
# GaussianProcessClassifier 의 정답률 :  [0.728      0.864      0.816      0.81451613 0.84677419]
# GaussianProcessClassifier 의 cross_val_predict ACC :  0.8246268656716418
# GradientBoostingClassifier 의 정답률 :  [0.752      0.864      0.808      0.82258065 0.79032258]
# GradientBoostingClassifier 의 cross_val_predict ACC :  0.7947761194029851
# HistGradientBoostingClassifier 의 정답률 :  [0.8        0.88       0.776      0.83064516 0.7983871 ]
# HistGradientBoostingClassifier 의 cross_val_predict ACC :  0.7985074626865671
# KNeighborsClassifier 의 정답률 :  [0.72       0.808      0.784      0.82258065 0.83870968]
# KNeighborsClassifier 의 cross_val_predict ACC :  0.7910447761194029
# LabelPropagation 의 정답률 :  [0.744      0.84       0.808      0.82258065 0.81451613]
# LabelPropagation 의 cross_val_predict ACC :  0.7985074626865671
# LabelSpreading 의 정답률 :  [0.744      0.848      0.808      0.82258065 0.81451613]
# LabelSpreading 의 cross_val_predict ACC :  0.8059701492537313
# LinearDiscriminantAnalysis 의 정답률 :  [0.744      0.848      0.824      0.82258065 0.83870968]
# LinearDiscriminantAnalysis 의 cross_val_predict ACC :  0.7910447761194029
# LinearSVC 의 정답률 :  [0.744      0.856      0.8        0.83064516 0.83870968]
# LinearSVC 의 cross_val_predict ACC :  0.7947761194029851
# LogisticRegression 의 정답률 :  [0.72       0.856      0.792      0.80645161 0.82258065]
# LogisticRegression 의 cross_val_predict ACC :  0.7947761194029851
# LogisticRegressionCV 의 정답률 :  [0.768      0.84       0.792      0.82258065 0.83870968]
# LogisticRegressionCV 의 cross_val_predict ACC :  0.8059701492537313
# MLPClassifier 의 정답률 :  [0.768      0.848      0.792      0.83064516 0.83870968]
# MLPClassifier 의 cross_val_predict ACC :  0.7985074626865671
# MultinomialNB 의 정답률 :  [0.696      0.808      0.816      0.80645161 0.83870968]
# MultinomialNB 의 cross_val_predict ACC :  0.7798507462686567
# NearestCentroid 의 정답률 :  [0.672      0.848      0.736      0.74193548 0.81451613]
# NearestCentroid 의 cross_val_predict ACC :  0.753731343283582
# NuSVC 의 정답률 :  [0.72       0.872      0.832      0.79032258 0.83870968]
# NuSVC 의 cross_val_predict ACC :  0.8171641791044776
# PassiveAggressiveClassifier 의 정답률 :  [0.768      0.856      0.784      0.83064516 0.68548387]
# PassiveAggressiveClassifier 의 cross_val_predict ACC :  0.7425373134328358
# Perceptron 의 정답률 :  [0.648      0.752      0.76       0.79032258 0.68548387]
# Perceptron 의 cross_val_predict ACC :  0.746268656716418
# QuadraticDiscriminantAnalysis 의 정답률 :  [0.76       0.848      0.808      0.79032258 0.82258065]
# QuadraticDiscriminantAnalysis 의 cross_val_predict ACC :  0.7238805970149254
# RadiusNeighborsClassifier 의 정답률 :  [0.704      0.816      0.824      0.79032258 0.83870968]
# RadiusNeighborsClassifier 의 cross_val_predict ACC :  0.7910447761194029
# RandomForestClassifier 의 정답률 :  [0.752      0.848      0.808      0.7983871  0.83064516]
# RandomForestClassifier 의 cross_val_predict ACC :  0.7723880597014925
# RidgeClassifier 의 정답률 :  [0.736      0.848      0.824      0.80645161 0.83870968]
# RidgeClassifier 의 cross_val_predict ACC :  0.7910447761194029
# RidgeClassifierCV 의 정답률 :  [0.736      0.848      0.824      0.80645161 0.83870968]
# RidgeClassifierCV 의 cross_val_predict ACC :  0.7798507462686567
# SGDClassifier 의 정답률 :  [0.752      0.824      0.8        0.81451613 0.79032258]
# SGDClassifier 의 cross_val_predict ACC :  0.6380597014925373
# SVC 의 정답률 :  [0.744      0.856      0.864      0.81451613 0.82258065]
# SVC 의 cross_val_predict ACC :  0.8171641791044776