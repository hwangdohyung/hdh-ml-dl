import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터 
x = np.array([range(10), range(21, 31), range(201, 211)])
# print(range(10)) # range: 범위, 거리 0~10까지의 정수형 숫자 
# for i in range(10):          # for :반복하라 
#     print(i)
print(x.shape)
x = np.transpose(x)
print(x.shape)

y = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
              [9,8,7,6,5,4,3,2,1,0]])  #.output 값 3개 

y = np.transpose(y)
print(y.shape)

#2.모델구성
model = Sequential()
model.add(Dense(50, input_dim=3)) 
model.add(Dense(40, )) 
model.add(Dense(30, )) 
model.add(Dense(40, )) 
model.add(Dense(40, )) 
model.add(Dense(10, ))
model.add(Dense(3, )) # output 3개!

#3.컴파일,훈련
model.compile(loss='mae', optimizer='adam') #음수가 나와서 mae
model.fit(x, y, epochs=500, batch_size=1)

#4.평가,예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([[9, 30, 210]]) 
print('[9, 30, 210]의 예측값 : ', result)

# loss :  0.21538448333740234
# [9, 30, 210]의 예측값 :  [[10.077459    1.9915777   0.11338423]]