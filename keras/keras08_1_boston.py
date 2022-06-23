import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# print(x)
# print(y)
# print(x.shape, y.shape) #(506, 13) (506, )
# print(datasets.feature_names)
# print(datasets.DESCR)

#[실습] 아래를 완성할것
#1.train 0.7
#2.R2 0.8 이상 

#2.모델구성

model = Sequential()
model.add(Dense(4, input_dim=13)) 
model.add(Dense(2))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))

#3.컴파일,훈련
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=66)
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1)

#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test) #훈련시키지 않은 부분을 평가 해야 되기 때문에 x_test.

#R2결정계수(성능평가지표)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 

# loss :  17.892765045166016
# r2스코어 :  0.7834254230300071
