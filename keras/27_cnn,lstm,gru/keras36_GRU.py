from lightgbm import early_stopping
import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,SimpleRNN,LSTM,GRU
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
#1.데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array(([1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]))
y = np.array([4,5,6,7,8,9,10])

print(x.shape,y.shape)
x = x.reshape(7, 3, 1) 
print(x.shape)  #(7,)


#2.모델구성
model = Sequential()
model.add(GRU(200,activation='relu', input_shape=(3,1)))  # LSTM의 forget gate와 input gate를 통합하여 하나의 update gate를 만든다. output gate는 없어지고 reset gate로 대체
model.add(Dense(100, activation= 'relu'))                #cell state 와 hidden state를 통합한다.
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(1))

model.summary()


# GRU    units : 10 -> 3*10* (1+1+10) = 360
# model.add(GRU(10,activation='relu', input_shape=(3,1)))
# SimpleRnn의 3배 gate가 줄었기 때문!
# 숫자 3의 의미는 hidden state, reste gate, update gate
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# gru (GRU)                    (None, 10)                360
# _________________________________________________________________
# dense (Dense)                (None, 100)               1100
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               10100
# _________________________________________________________________
# dense_2 (Dense)              (None, 100)               10100
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 101
# =================================================================
# Total params: 21,761
# Trainable params: 21,761
# Non-trainable params: 0
# ___________________________________________




#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
earlyStopping = EarlyStopping(monitor= 'loss',patience=20, mode='min',restore_best_weights=True,verbose=1)
model.fit(x,y, epochs=500, batch_size=32,verbose=1)

#4.평가,예측 
loss = model.evaluate(x,y)
y_pred = np.array([8,9,10]).reshape(1,3,1)  # [[[8],[9],[10]]]  
result = model.predict(y_pred)  #모델은 3차원을 원한다.
print('loss : ', loss)
print('result : ', result)
