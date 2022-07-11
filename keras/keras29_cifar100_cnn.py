import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
print(x_train.shape)
import numpy as np
print(np.unique(y_train,return_counts=True))
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2.모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(32,32,3))) 
model.add(MaxPooling2D())
model.add(Conv2D(7, (2,2), padding='valid', activation='relu'))
model.add(Conv2D(7, (2,2), padding='valid', activation='relu'))
model.add(Conv2D(7, (2,2), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) 
model.summary()
#3.컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=610)

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
print(acc)

0.5312