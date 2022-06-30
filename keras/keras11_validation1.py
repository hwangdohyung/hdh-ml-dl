import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1.데이터 # 1~ 16
x_train = np.array(range(1,11))#훈련 
y_train = np.array(range(1,11))#훈련
x_test =  np.array([11,12,13])#evaluate에서 씀 
y_test =  np.array([11,12,13])#evaluate에서 씀
x_val =  np.array([14,15,16])# 검증 
y_val =  np.array([14,15,16])# 검증 

#2.모델구성
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(5,))
model.add(Dense(5,))
model.add(Dense(5,))
model.add(Dense(1,))

#3.컴파일,훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 100, batch_size=10,
          validation_data=(x_val,y_val))

#4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss : ' , loss)

result = model.predict([17])
print('17의 예측값 : ', result)

#훈련 loss보다 검증 loss값이 더 떨어질 수 밖에 없다. (통상 그렇다,항상 그런것은 아님)



