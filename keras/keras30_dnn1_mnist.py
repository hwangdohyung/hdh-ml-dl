from keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
#1.데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
print(x_train.shape)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. 모델구성
model =Sequential()
# model.add(Dense(64, input_shape = (28*28,)))
model.add(Dense(64,activation='relu', input_shape = (784,)))   #위와 동일
model.add(Dropout(0.2))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax')) 

# 3.컴파일,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam')

earlyStopping = EarlyStopping(monitor= 'val_loss',patience=10,mode='min',restore_best_weights=True,verbose=1)

# filepath = './_ModelCheckpoint/k24/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1,
#                       save_best_only=True,filepath= "".join([filepath ,'boston',date,'_', filename]))

hist = model.fit(x_train,y_train,epochs=50,batch_size=32,verbose=1,validation_split= 0.2,callbacks=[earlyStopping])

#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) 
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 

y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)
acc = accuracy_score(y_test, y_predict) 
print('acc스코어: ', acc)

# loss :  0.22856928408145905
# r2스코어 :  0.9005785671115214
# acc스코어:  0.9435

# (kernel_size * channels +bias) *filters = summary Param 갯수(CNN 모델)
# 
#CNN input: none, row, column , channel 
#    output: none, row , column, filter

#DNN input: none, input_dim 
#    output: none, unit(output)

# Dense layer로 변환할때
#- 4차원 ->2차원으로 바꾸는것 데이터를 쭉 늘인다.(reshape) 순서는 바뀌지 않음. (n,4,3,2)-> (n,24) 
