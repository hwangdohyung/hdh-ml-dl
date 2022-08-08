import numpy as np 
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Input,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import datetime
import pandas as pd

#1.데이터 
datasets = fetch_covtype()
x= datasets.data
y= datasets.target


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=True,
                                                     random_state=58)

# minmax , standard
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 

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

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)
model5.fit(x_train,y_train)
model6.fit(x_train,y_train)

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

