import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

#1.데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x)
# print(y)
# print(x.shape, y.shape) #(20640, 8) (20640,)
# print(datasets.feature_names)
# print(datasets.DESCR)

#2.모델구성
model = Sequential()
model.add(Dense(4, input_dim=8)) 
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(6))
model.add(Dense(2))
model.add(Dense(5))
model.add(Dense(1))

#3.컴파일,훈련
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100)
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=400, batch_size=100)

#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

#R2결정계수(성능평가지표)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 

# loss :  0.6198102831840515
# r2스코어 :  0.532244021900772

