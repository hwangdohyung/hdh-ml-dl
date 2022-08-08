from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler


#1.데이터
datasets = load_iris()

x = datasets['data']
y = datasets['target']
print(x.shape, y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)


scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
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


# SCV :  0.9777777777777777
# Perceptron :  0.6666666666666666
# LogisticRegression :  0.9555555555555556
# KNeighborsClassifier :  0.9555555555555556
# DecisionTreeClassifier :  0.9555555555555556
# RandomForestClassifier :  0.9777777777777777
import sklearn as sk
print(sk.__version__)

