import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,LSTM,Conv1D
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping

#1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
print(x_train.shape)
import numpy as np
print(np.unique(y_train,return_counts=True))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape,x_test.shape)
x_train = x_train.reshape(60000,28*28,1)
x_test = x_test.reshape(10000,28*28,1)

#2.모델구성
model = Sequential()
model.add(Conv1D(8,2,activation='relu', input_shape=(28*28,1))) 
model.add(Flatten())
model.add(Dense(6,activation= 'relu'))
model.add(Dense(4,activation= 'relu'))
model.add(Dense(2,activation= 'relu'))
model.add(Dense(10,activation='softmax'))

#3.컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam')
earlyStopping = EarlyStopping(monitor = 'loss',patience=10,mode='min',restore_best_weights=True,verbose=1)
model.fit(x_train, y_train, epochs=10, batch_size=512)


print(y_test)
#4.평가 훈련
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_test)
print(y_predict)
y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc스코어: ', acc)

#CNN
# 0.9809

#LSTM
# acc스코어:  0.1135

#Conv1D
# acc스코어:  0.7121