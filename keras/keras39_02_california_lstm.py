import numpy as np 
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Input,Dropout,LSTM
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
#1.데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100)

# minmax , standard
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape,x_test.shape)

x_train = x_train.reshape(14447,8,1)
x_test = x_test.reshape(6193,8,1)

#2.모델구성
model = Sequential()
model.add(LSTM(200,activation='relu', input_shape=(8,1))) 
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(1))


import datetime
date = datetime.datetime.now(   )
date = date.strftime('%m%d_%H%M')


#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')

earlyStopping = EarlyStopping(monitor= 'val_loss',patience=10,mode= 'min', restore_best_weights=True,verbose=1)

# filepath = './_ModelCheckpoint/k24/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor= 'val_loss',mode='auto',verbose=1,save_best_only=True,
#                       filepath ="".join([filepath,'california',date, '_',filename]))

hist = model.fit(x_train, y_train, epochs=100, batch_size=128,
         validation_split= 0.2,callbacks= [earlyStopping])

print(y_test.shape)
# y_test = y_test.reshape(6193,1,1)
#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 

#DNN
# loss :  0.2756466567516327
# r2스코어 :  0.7919760853563619

#LSTM
# loss :  0.4285742938518524
# r2스코어 :  0.6765652713217769
