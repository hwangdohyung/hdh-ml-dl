import tensorflow as tf 
import numpy as np 
import pandas as pd
from tensorflow.python.keras.layers import Dense,Flatten,Conv2D,Dropout,Conv2DTranspose,Input
from tensorflow.python.keras.models import Model,Sequential
from sklearn.model_selection import train_test_split 
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint


#1.data 
x = [1,2,3,4,5,6,7,]
y = [10,20,30,40,50,60,70]

#2.model
model = Sequential()
model.add(Dense(30,input_dim=1))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(1,activation='linear'))

#3.compile,fit
model.compile(loss='mse',optimizer='adam') 
model.fit(x,y,epochs=500,verbose=1)

#4.evaluate,predict
loss = model.evaluate(x,y)
result = model.predict([8])
print('loss: ' ,loss)
print('result: ' , result)



