import numpy as np
from sklearn import metrics 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LSTM,GRU
from tensorflow.python.keras.callbacks import EarlyStopping

#1.데이터

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
# x_predict = np.array([50,60,70])           #i wanna 80

print(x.shape,y.shape)
x= x.reshape(13,3,1)
print

print(x.shape,y.shape)

#2.모델구성
model = Sequential()
model.add(GRU(200,activation='relu', input_shape=(3,1))) 
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
earlyStopping = EarlyStopping(monitor= 'loss',patience=20, mode='min',restore_best_weights=True,verbose=1)
model.fit(x,y, epochs=500, batch_size=32,verbose=1,callbacks=earlyStopping)

#4.평가,예측 
loss = model.evaluate(x,y)
y_pred = np.array([50,60,70]).reshape(1,3,1)  
result = model.predict(y_pred)  #모델은 3차원을 원한다. 
print('loss : ', loss)
print('result : ', result)

############# rnn 데이터 자르는 함수 ##############
# data = ([1,2,3,4,5,6,7,8,9,10,11,12,13,20,30,40,50,60,70])

#1. 데이터

# data = np.array([1,2,3,4,5,6,7,8,9,10])

# def split_xy1(dataset, time_steps):
#   x, y = list(), list()
#   for i in range(len(dataset)):
#     end_number = i + time_steps
#     if end_number > len(dataset) - 1:
#       break
#     tmp_x, tmp_y = dataset[i:end_number], dataset[end_number]
#     x.append(tmp_x)
#     y.append(tmp_y)
#   return np.array(x), np.array(y)

# x, y = split_xy1(data, 3)     #### 여기서 3부분에 컬럼 수 넣는다.

