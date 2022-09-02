from lightgbm import early_stopping
import numpy as np 
from tensorflow.keras.models import Sequential               #전부 구버전으로 해줘야 함                
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM     #""
from sklearn.model_selection import train_test_split          
from tensorflow.keras.callbacks import EarlyStopping         #""
from tensorflow.keras.layers import Bidirectional            # version 문제 

#1.데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array(([1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9])) 
y = np.array([4,5,6,7,8,9,10])

print(x.shape,y.shape)
x = x.reshape(7, 3, 1)
print(x.shape)

#2.모델구성
model = Sequential()
model.add(SimpleRNN(200,activation='relu', input_shape=(3,1),return_sequences=True)) 
model.add(Bidirectional(SimpleRNN(100))) 
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(1))
model.summary()


# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 3, 10)             120

#  bidirectional (Bidirectiona  (None, 10)               160
#  l)

#  dense (Dense)               (None, 3)                 33

#  dense_1 (Dense)             (None, 1)                 4

# =================================================================
# Total params: 317
# Trainable params: 317
# Non-trainable params: 0
# _________________________________________________________________
# (5*5) + (5*10) + (5*1) *2 = 160



#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=600, batch_size=32,verbose=1)

#4.평가,예측 
loss = model.evaluate(x,y)
y_pred = np.array([8,9,10]).reshape(1,3,1)  
result = model.predict(y_pred) 
print('result : ', result)

# result :  [[10.8262005]]

 