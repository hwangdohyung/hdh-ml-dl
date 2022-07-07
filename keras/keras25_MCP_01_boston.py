#12개 만들고... 최적의 weigth 찾기!
import numpy as np 
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint 

#1.데이터
datasets = load_boston()
x,y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)
                                                                                                                         
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)                 
x_test = scaler.transform(x_test)

#2.모델구성
model = Sequential()
model.add(Dense(64, input_dim=13)) 
model.add(Dense(32,activation= 'relu'))
model.add(Dense(32,activation= 'relu'))
model.add(Dense(32,activation= 'relu'))
model.add(Dense(16,activation= 'relu'))
model.add(Dense(1))

#시간
import datetime
date = datetime.datetime.now()
print(date)                       #2022-07-07 17:23:32.783752
date = date.strftime('%m%d_%H%M') #0707_1723
print(date)


# 3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')

earlyStopping = EarlyStopping(monitor= 'val_loss',patience=10,mode='min',restore_best_weights=True,verbose=1)

filepath = './_ModelCheckpoint/k24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1,
                      save_best_only=True,filepath= "".join([filepath ,'boston',date,'_', filename])) #문자와 문자를 더하면 이어진다 ,""join = 안에 모든 문자열 합치겠다.

hist = model.fit(x_train,y_train,epochs=100,batch_size=1,verbose=1,validation_split= 0.2,callbacks=[earlyStopping, mcp])


#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) 
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 

