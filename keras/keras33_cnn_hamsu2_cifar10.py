import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense, Conv2D, Dropout,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


x_train = x_train.reshape(50000,3072)
x_test = x_test.reshape(10000,3072)

import numpy as np
print(np.unique(y_train,return_counts=True))

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
 

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#2.모델구성
input1= Input(shape=(3072,))
dense1 = Dense(40,activation= 'relu')(input1)
dense2 = Dense(50,activation= 'relu')(dense1)
dense3 = Dense(50,activation= 'relu')(dense2)
dense4 = Dense(50,activation= 'relu')(dense3)
dense5 = Dense(50,activation= 'relu')(dense4)
output1= Dense(10,activation='softmax')(dense5)
model = Model(inputs =input1,outputs=output1)

#3.컴파일 훈련

earlyStopping = EarlyStopping(monitor= 'val_loss',patience=10,mode='min',restore_best_weights=True,verbose=1)


model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=150,validation_split=0.2,verbose = 1,callbacks= earlyStopping)


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

# acc스코어:  0.5069