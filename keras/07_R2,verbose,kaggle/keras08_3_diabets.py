import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes 

#1.데이터
datasets = load_diabetes()
x = datasets.data 
y = datasets.target 

# print(x)
# print(y)
# print(x.shape, y.shape) #(442, 10) (442,)
# print(datasets.feature_names) 열 특징의 이름 
# print(datasets.DESCR) 특징 설명 

#[실습]
#R2 0.62 이상

#2.모델구성
model = Sequential()
model.add(Dense(4,input_dim=10))
model.add(Dense(20))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

#3.컴파일,훈련
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=72)
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1)

#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

#R2결정계수(성능평가지표)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 

#loss :  2267.363037109375
# r2스코어 :  0.6215301607936815

