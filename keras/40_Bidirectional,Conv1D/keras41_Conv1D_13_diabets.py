import numpy as np
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input,Dropout,LSTM,Conv1D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes 
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

#1.데이터
datasets = load_diabetes()
x = datasets.data 
y = datasets.target 
print(datasets)
print(datasets.feature_names)
print(datasets.DESCR)
print('y의 라벨값 : ', np.unique(y,return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=72)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 

print(x_train.shape,x_test.shape)
x_train = x_train.reshape(309,10,1)
x_test = x_test.reshape(133,10,1)

#2.모델구성
model = Sequential()
model.add(Conv1D(200,2,activation='relu', input_shape=(10,1))) 
model.add(Flatten())
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(1))


import datetime
date = datetime.datetime.now()
date =date.strftime('%m%d_%H%M')

#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
earlyStopping = EarlyStopping(monitor= 'val_loss',patience=10,mode='min', restore_best_weights=True,verbose=1)

# filepath ='./_ModelCheckpoint/k24/'
# filename ='{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp= ModelCheckpoint(monitor = 'val_loss',mode='auto',save_best_only=True,verbose=1,
#                      filepath = "".join([filepath,'diabets',date,'_',filename]))

hist = model.fit(x_train, y_train, epochs=200, batch_size=128,validation_split=0.2,callbacks=[earlyStopping])

#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

#R2결정계수(성능평가지표)
from sklearn.metrics import r2_score,accuracy_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 

#DNN
# loss :  2396.525390625
# r2스코어 :  0.59997033090829

#LSTM
# loss :  4134.39599609375
# r2스코어 :  0.3098837384251836

#Conv1D
# loss :  2186.356201171875
# r2스코어 :  0.6350518373197113