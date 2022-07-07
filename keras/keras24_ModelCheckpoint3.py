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
model.add(Dense(16,activation= 'relu'))
model.add(Dense(1))

# 3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')

earlyStopping = EarlyStopping(monitor= 'val_loss',patience=10,mode='min',restore_best_weights=True,verbose=1)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1,
                      save_best_only=True,filepath='./_ModelCheckpoint/keras24_ModelCheckpoint3.hdf5')
#monitor한것에 대한 mode가 가장 좋은것이 저장됨. patience는 필요x 

hist = model.fit(x_train,y_train,epochs=1000,batch_size=1,verbose=1,validation_split= 0.2,callbacks=[earlyStopping, mcp],)

model.save('./_save/keras24_3_save_model.h5')


#4.평가,예측
print("======================== 1. 기본 출력 ==========================")
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) 
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 


print("======================== 2.load_model 출력 ==========================")
model2 = load_model('./_save/keras24_3_save_model.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss2 : ', loss2)

y_predict2 = model2.predict(x_test) 
r2 = r2_score(y_test, y_predict2) 
print('r2스코어 : ' , r2) 


print("======================== 3.ModelCheckpoint 출력 ==========================")
model3 = load_model('./_ModelCheckpoint/keras24_ModelCheckpoint3.hdf5')
loss3 = model3.evaluate(x_test, y_test)
print('loss2 : ', loss2)

y_predict3 = model3.predict(x_test) 
r2 = r2_score(y_test, y_predict3) 
print('r2스코어 : ' , r2) 

# ======================== 1. 기본 출력 ==========================
# 4/4 [==============================] - 0s 0s/step - loss: 10.6789
# loss :  10.67891788482666
# r2스코어 :  0.8722356768551883
# ======================== 2.load_model 출력 ==========================
# 4/4 [==============================] - 0s 776us/step - loss: 10.6789
# loss2 :  10.67891788482666
# r2스코어 :  0.8722356768551883
# ======================== 3.ModelCheckpoint 출력 ==========================
# 4/4 [==============================] - 0s 666us/step - loss: 10.6789
# loss2 :  10.67891788482666
# r2스코어 :  0.8722356768551883

