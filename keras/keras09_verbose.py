from tabnanny import verbose
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

#2.모델구성
model = Sequential()
model.add(Dense(4, input_dim=13)) 
model.add(Dense(2))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))

import time    #시간 
#3.컴파일,훈련
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=66)
model.compile(loss='mse', optimizer='adam')
start_time = time.time() #현재 시간을 알려준다.
print(start_time)            #1656032968.308002
model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=1) #verbose:(장황한) 0으로 하면 훈련과정을 보여주지 않는다. 성능의 차이는x , 속도차이o 


end_time = time.time() - start_time

print("걸린시간: ", end_time)

"""
verbose 0 걸린시간 : 10.16734004020691 / 출력없다.
verbose 1 걸린시간 : 12.461289882659912/ 잔소리 많다.
verbose 2 걸린시간 : 10.539838790893555/ 프로그래스바 없다.
verbose 3 걸린시간 : 10.07824993133545 / epoch만 나온다.
4,5,6 이상은 3과 동일

"""