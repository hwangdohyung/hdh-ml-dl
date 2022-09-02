from lightgbm import early_stopping
import numpy as np
from tensorflow.python.keras.models import Sequential                            
from tensorflow.python.keras.layers import Dense,SimpleRNN,LSTM,Conv1D,Flatten
from sklearn.model_selection import train_test_split          
from tensorflow.python.keras.callbacks import EarlyStopping         

#1.데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array(([1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9])) 
y = np.array([4,5,6,7,8,9,10])

print(x.shape,y.shape)
x = x.reshape(7, 3, 1)
print(x.shape)

#2.모델구성
model = Sequential()
# model.add(LSTM(10,activation='relu', input_shape=(3,1))) 
model.add(Conv1D(10,2, input_shape=(3,1)))
model.add(Flatten())
model.add(Dense(3,activation='relu'))
model.add(Dense(1))
model.summary()

'''
#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=600, batch_size=32,verbose=1)

#4.평가,예측 
loss = model.evaluate(x,y)
y_pred = np.array([8,9,10]).reshape(1,3,1)  
result = model.predict(y_pred) 
print('result : ', result)

# result :  [[10.8262005]]

'''