import pandas as pd 
import numpy as np 
import os 

path = 'D:\study_data\_data\dacon_chung/'

x = pd.read_csv(path + 'train_input/CASE_01.csv',index_col=0)[:1440]

print(x.shape) #(1440, 37)
# x.to_csv('D:\study_data\_temp/x.csv')
y = pd.read_csv(path + 'train_target/CASE_01.csv',index_col=0)[:1]

print(y)
from xgboost import XGBRegressor
from tensorflow.python.keras.layers import Dense,SimpleRNN,LSTM
from tensorflow.python.keras.models import Sequential
x = x.to_numpy()

model = XGBRegressor()
model.fit(x,y)
print("gg : ",  model.score(x,y))


# model = Sequential()
# model.add(LSTM(20,activation='relu', input_shape=(37,1))) #input_shape 행무시 #dense 로 넘어갈 때 2차원으로 던져줌. 바로 dense로 받는거 가능(flatten x)
# model.add(Dense(10, activation= 'relu'))
# model.add(Dense(10, activation= 'relu'))
# model.add(Dense(10, activation= 'relu'))
# model.add(Dense(1))

# #3.컴파일,훈련
# model.compile(loss='mse', optimizer='adam')
# # earlyStopping = EarlyStopping(monitor= 'val_loss',patience=60, mode='min',restore_best_weights=True,verbose=1)
# model.fit(x,y, epochs=500, batch_size=1,verbose=1)

# #4.평가,예측 
# loss = model.evaluate(x,y)
# print('loss : ', loss)


