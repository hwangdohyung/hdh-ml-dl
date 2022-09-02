import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,Dropout,Input

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = x_train.reshape(60000,28*28*1)
x_test = x_test.reshape(10000,28*28*1)
print(x_train.shape)

import numpy as np
print(np.unique(y_train,return_counts=True))
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2.모델구성
input1= Input(shape=(28*28*1))
dense1 = Dense(40,activation='relu')(input1)
dense2 = Dense(40,activation='relu')(dense1)
dense3 = Dense(40,activation='relu')(dense2)
dense4 = Dense(40,activation='relu')(dense3)
dense5 = Dense(40,activation='relu')(dense4)
output1= Dense(10,activation='softmax')(dense5)
model= Model(inputs=input1,outputs=output1)

#3.컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=20)

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
print(':acc스코어 ', acc)

# :acc스코어  0.7355

#함수
# :acc스코어  0.8635