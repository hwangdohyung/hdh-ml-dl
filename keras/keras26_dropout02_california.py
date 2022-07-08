import numpy as np 
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Input,Dropout
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

#2.모델구성
input1 = Input(shape = (8,))
dense1 = Dense(50,activation= 'relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(50,activation= 'relu')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(50,activation= 'relu')(drop2)
dense4 = Dense(50,activation= 'relu')(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)

import datetime
date = datetime.datetime.now(   )
date = date.strftime('%m%d_%H%M')


#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')

earlyStopping = EarlyStopping(monitor= 'val_loss',patience=30,mode= 'min', restore_best_weights=True,verbose=1)

# filepath = './_ModelCheckpoint/k24/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor= 'val_loss',mode='auto',verbose=1,save_best_only=True,
#                       filepath ="".join([filepath,'california',date, '_',filename]))

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
         validation_split= 0.2,callbacks= [earlyStopping])


#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 

# loss :  0.2756466567516327
# r2스코어 :  0.7919760853563619