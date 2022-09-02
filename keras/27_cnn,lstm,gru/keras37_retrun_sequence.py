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

x= x.reshape(13,3,1)

#2.모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM,Dense

print(x.shape,y.shape)
model = Sequential()
model.add(LSTM(100,activation='relu', return_sequences=True, input_shape = (3,1))) # return_sequences True= 한차원 늘려서 던져준다. #(n,3,1) -> (n,3,10)
print(x.shape)
model.add(LSTM(100, input_shape=(3,100)))
model.add(Dense(100, activation='relu')) 
model.add(Dense(100, activation='relu')) 
model.add(Dense(100, activation='relu')) 
model.add(Dense(1))

model.summary()

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

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 3, 10)             480           
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 5)                 320
# _________________________________________________________________
# dense (Dense)                (None, 1)                 6
# =================================================================
# Total params: 806
# Trainable params: 806
# Non-trainable params: 0

#LSTM 2개 엮은거 테스트해보고


