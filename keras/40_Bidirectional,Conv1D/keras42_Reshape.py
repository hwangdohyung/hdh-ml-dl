import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Conv1D,LSTM,Reshape #
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

#1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
print(x_train.shape)
import numpy as np
print(np.unique(y_train,return_counts=True))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train[:5])
#acc 0.98이상
#cnn 3개이상

#2.모델구성
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(28,28,1))) 
model.add(MaxPooling2D())                                               # (N,14,14,64)
model.add(Conv2D(32, (3,3)))                                            # (N,12,12,32)
model.add(Conv2D(7, (3,3)))                                             # (N,10,10,7)   
model.add(Reshape(target_shape = (-1,)))                                # (N,700) ************************
# model.add(Flatten())                                                  # (N,700)
model.add(Dense(100, activation='relu'))                                # (N,100)
model.add(Reshape(target_shape = (100,1)))                              # (N,100,1)layer reshape 연산은 하지 않는다,모양만 바꿔줌 ,순서와 내용은 안바뀜. Flatten과 동일
model.add(Conv1D(10,kernel_size = 3))                                   # (N,98,10) #padding 안했기 때문 양쪽 끝 2칸 잘림 98
model.add(LSTM(16))                                                     # (N,16)
model.add(Dense(32, activation='relu'))                                 # (N,32)                    
model.add(Dense(10, activation='softmax'))                              # (N,10)
model.summary()

'''
#3.컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=20)

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

# 0.9809
#
'''