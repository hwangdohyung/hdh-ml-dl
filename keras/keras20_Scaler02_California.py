import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

#1.데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100)

# minmax , standard
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 

#2.모델구성
model = Sequential()
model.add(Dense(50, input_dim=8)) 
model.add(Dense(50,activation ='relu'))
model.add(Dense(50,activation ='relu'))
model.add(Dense(50,activation ='relu'))
model.add(Dense(50,activation ='relu'))
model.add(Dense(50,activation ='relu'))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping

earlyStopping = EarlyStopping(monitor= 'val_loss',patience=30,mode= 'min', restore_best_weights=True,verbose=1)



hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
         validation_split= 0.2,callbacks= earlyStopping)

#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 

#minmax
# loss :  0.24442075192928314
# r2스코어 :  0.8155415129463858

#standard
# loss :  0.25456172227859497
# r2스코어 :  0.807888387977042

#Maxabs
# loss :  0.33512821793556213
# r2스코어 :  0.7470867911156376

#robust
# loss :  0.2823280990123749
# r2스코어 :  0.7869337806171974

#none
# loss :  0.4566934108734131
# r2스코어 :  0.6553444555271739