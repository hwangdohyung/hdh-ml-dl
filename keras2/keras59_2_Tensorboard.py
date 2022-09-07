import numpy as np
from keras.datasets import mnist,cifar100
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input,Dropout
import keras
import tensorflow as tf 
from keras.layers import GlobalAveragePooling2D
from sklearn.metrics import accuracy_score

# 1.데이터
(x_train,y_train),(x_test,y_test)  = mnist.load_data()

x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28,28,1)
x_test = x_test.reshape(10000, 28,28,1)

# from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# 2.모델 
activation = 'relu'
drop = 0.2

inputs = Input(shape= (28, 28, 1), name= 'input')
x = Conv2D(64, (2, 2), padding='valid', activation=activation, name= 'hidden1')(inputs)
x = Dropout(drop)(x)
x = MaxPool2D()(x)
x = Conv2D(32, (3, 3), padding='valid', activation=activation, name= 'hidden3')(x)
x = Dropout(drop)(x)
x = GlobalAveragePooling2D()(x) 
x = Dense(100, activation=activation, name='hidden4')(x)
x = Dropout(drop)(x)
outputs = Dense(10, activation='softmax', name='outputs')(x)
model = Model(inputs=inputs, outputs=outputs)

# 3.컴파일,훈련
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,TensorBoard
es = EarlyStopping(monitor='val_loss',patience=15, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1,
                              factor=0.5)

tb = TensorBoard(log_dir='D:\study_data\_tensorboard_log\_graph/', histogram_freq=0,
                 write_graph=True, write_images=True)
#실행방법 : tensorboard --logdir=. (경로)
# http://localhost:6006 
# 또는
# http://127.0.0.1:6006

from keras.optimizers import Adam 
learning_rate= 0.01
optimizer= Adam(learning_rate=learning_rate)

model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,
              metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=100, batch_size=32, verbose=1,
                 callbacks=[es,reduce_lr,tb],validation_split=0.2)

loss, acc = model.evaluate(x_test,y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('acc : ', round(acc, 4))

import matplotlib.pyplot as plt
############# 시각화 ###############
#1. 
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

#2.
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])

plt.show()



