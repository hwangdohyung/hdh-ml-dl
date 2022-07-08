import numpy as np
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
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

# # minmax , standard
# scaler = MinMaxScaler()
#  scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 


#2.모델구성
# model = Sequential()
# model.add(Dense(40,input_dim=10))
# model.add(Dense(40))
# model.add(Dense(40))
# model.add(Dense(40))
# model.add(Dense(40))
# model.add(Dense(1))

input1= Input(shape= (10,))
dense1 =Dense(40,activation='relu')(input1)
dense2 =Dense(40,activation='relu')(dense1)
dense3 =Dense(40,activation='relu')(dense2)
dense4 =Dense(40,activation='relu')(dense3)
dense5 =Dense(40,activation='relu')(dense4)
dense6 =Dense(40,activation='relu')(dense5)
output1= Dense(1)(dense6)
model = Model(inputs=input1,outputs=output1)

#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor= 'val_loss',patience=50,mode='min', restore_best_weights=True,verbose=1)


hist = model.fit(x_train, y_train, epochs=1000, batch_size=10,validation_split=0.2,callbacks=earlyStopping)

model.save("./_save/keras23_09_save_diabets.h5")

#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

#R2결정계수(성능평가지표)
from sklearn.metrics import r2_score,accuracy_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 

# loss :  2311.641845703125
# r2스코어 :  0.614139058752844