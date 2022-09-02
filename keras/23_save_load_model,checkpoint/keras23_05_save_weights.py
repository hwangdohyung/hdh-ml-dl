from tabnanny import verbose
import numpy as np 
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,StandardScaler,RobustScaler


#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=66)

# minmax , standard ,maxabs , robust
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 


#2.모델구성
model = Sequential()
model.add(Dense(64, input_dim=13)) 
model.add(Dense(32,activation= 'relu'))
model.add(Dense(16,activation= 'relu'))
model.add(Dense(8,activation= 'relu'))
model.add(Dense(1))
model.summary()

# model.save("./_save/keras23_1_save_model.h5")
model.save_weights("./_save/keras23_5_save_weights1.h5") # 훈련전의 랜덤 가중치 쓰레기 저장됨

# 3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor= 'val_loss',patience=50,mode='min',restore_best_weights=True,verbose=1)

model.fit(x_train,y_train,epochs=100,batch_size=1,verbose=1,validation_split= 0.2,callbacks=earlyStopping)


model.save_weights("./_save/keras23_5_save_weights2.h5")#훈련뒤의 좋은 가중치가 저장됨



#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test) 

#R2결정계수(성능평가지표)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 

# load 하기전 
# loss :  12.8367919921875
# r2스코어 :  0.8464184996780377

# load 후
# loss :  12.8367919921875
# r2스코어 :  0.8464184996780377

