import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)


x_train = x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 
x_train = x_train.reshape(50000, 32,32, 3)
x_test = x_test.reshape(10000,32,32,3)
print(x_train.shape)
print(y_train.shape)
print(y_train[:5])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()


#2.모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(32,32,3))) 
model.add(MaxPooling2D())
model.add(Conv2D(7, (2,2), padding='valid', activation='relu'))
model.add(Conv2D(84, (2,2), padding='same', activation='relu'))
model.add(Conv2D(7, (2,2), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation='softmax')) 
model.summary()
#3.컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
ES = EarlyStopping(monitor='val_loss', patience=20, mode='min')
MCP = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                      filepath='./_ModelCheckPoint/k29/cifar100.hdf5')

model.fit(x_train, y_train, epochs=10, batch_size=600, callbacks=[ES,MCP], validation_split=0.05)

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
