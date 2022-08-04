from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler


#1.데이터
datasets = load_boston()

x = datasets['data']
y = datasets['target']


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)


#2.모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression,LinearRegression #logisticregression : regression 이 들어가지만 분류다!
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# model1 = LinearSVC()
# model2 = Perceptron()
model3 = LinearRegression()
model4 = KNeighborsRegressor()
model5 = DecisionTreeRegressor()
model6 = RandomForestRegressor()

#3.컴파일,훈련

# model1.fit(x_train,y_train)
# model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)
model5.fit(x_train,y_train)
model6.fit(x_train,y_train)

#4.평가,예측

# result1 = model1.score(x_test, y_test)
# print('SCV : ', result1)


# result2 = model2.score(x_test, y_test)
# print('Perceptron : ', result2)

result3 = model3.score(x_test, y_test)
print('LinearRegression : ', result3)


result4 = model4.score(x_test, y_test)
print('KNeighborsRegressor : ', result4)


result5 = model5.score(x_test, y_test)
print('DecisionTreeRegressor : ', result5)

result6 = model6.score(x_test, y_test)
print('RandomForestRegressor : ', result6)

# LinearRegression :  0.7567743451802437
# KNeighborsRegressor :  0.4823983034178225
# DecisionTreeRegressor :  0.6015304264071909
# RandomForestRegressor :  0.8332533937926487