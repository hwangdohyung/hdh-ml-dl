#mlp = 멀티레이어퍼셉트론
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1.데이터 
x = np.array([range(10)])
print(x.shape) #(1, 10)

x = np.transpose(x) #(10, 1)
print(x.shape)

y = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],      # y1,y2,y3 = wx+b
              [9,8,7,6,5,4,3,2,1,0]])  

y = np.transpose(y)
print(y.shape)


#2.모델구성
model = Sequential()
model.add(Dense(50, input_dim=1)) 
model.add(Dense(40, )) 
model.add(Dense(30, )) 
model.add(Dense(40, )) 
model.add(Dense(80, )) 
model.add(Dense(40, )) 
model.add(Dense(10, ))
model.add(Dense(3, )) 

#3.컴파일,훈련
model.compile(loss='mae', optimizer='adam') 
model.fit(x, y, epochs=500, batch_size=1)

#4.평가,예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([9]) 
print('[9]의 예측값 : ', result)

# loss :  0.08555112034082413
# [9]의 예측값 :  [[10.054097    1.6894062   0.09362863]]

