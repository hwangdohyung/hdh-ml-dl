import numpy as np
from sklearn import metrics 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

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
# model.add(Bidirectional(LSTM(200,return_sequences=True),input_shape=(3,1))) 
model.add(Bidirectional(LSTM(200),input_shape=(3,1))) 
# model.add(LSTM(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
earlyStopping = EarlyStopping(monitor= 'loss',patience=40, mode='min',restore_best_weights=True,verbose=1)
model.fit(x,y, epochs=500, batch_size=32,verbose=1,callbacks=earlyStopping)

#4.평가,예측 
loss = model.evaluate(x,y)
y_pred = np.array([50,60,70]).reshape(1,3,1)
result = model.predict(y_pred)  #모델은 3차원을 원한다. 
print('loss : ', loss)
print('result : ', result)

# loss :  3.7772770156152546e-05
# result :  [[80.23521]]

