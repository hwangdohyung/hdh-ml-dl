#https://blog.naver.com/qbxlvnf11/221369724614

import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)




x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)

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
ES = EarlyStopping(monitor='val_loss', patience=10, mode='min',restore_best_weights=True,verbose=1)

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=100,callbacks=[ES],verbose=1,validation_split=0.2)

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
print('acc스코어: ',acc)

# acc스코어:  0.5879
