import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input,Dropout
import keras
import tensorflow as tf 
from tensorflow.keras.layers import GlobalAveragePooling2D
# 1.데이터
(x_train,y_train),(x_test,y_test)  = mnist.load_data()

x_train = x_train.reshape(60000, 28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000, 28,28,1).astype('float32')/255.


from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# 2.모델 
activation = 'relu'
drop = 0.2
optimizer = 'adam'

inputs = Input(shape= (28, 28, 1), name= 'input')
x = Conv2D(64, (2, 2), padding='valid', activation=activation, name= 'hidden1')(inputs)
x = Dropout(drop)(x)
# x = Conv2D(64, (2, 2), padding='same', activation=activation, name= 'hidden2')(x)
# x = Dropout(drop)(x)
x = MaxPool2D()(x)
x = Conv2D(32, (3, 3), padding='valid', activation=activation, name= 'hidden3')(x)
x = Dropout(drop)(x)
x = GlobalAveragePooling2D()(x) 
# x = Flatten()(x)

x = Dense(100, activation=activation, name='hidden4')(x)
x = Dropout(drop)(x)
outputs = Dense(10, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
model.summary()


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

import time
start = time.time()
model.fit(x_train,y_train, epochs=5, validation_split=0.4,batch_size=128)
end = time.time()

from sklearn.metrics import accuracy_score


y_predict = model.predict(x_test)
print('걸린시간 : ', start-end)
print('acc : ', accuracy_score(y_test,y_predict))





    