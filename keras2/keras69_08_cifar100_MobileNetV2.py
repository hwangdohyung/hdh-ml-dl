
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense,Flatten,GlobalAveragePooling2D
from keras.applications import MobileNetV2
from keras.datasets import cifar10,cifar100
import tensorflow as tf 
tf.random.set_seed(123)


# 1.데이터
(x_train,y_train),(x_test,y_test)  = cifar100.load_data()

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32,32,3)
x_test = x_test.reshape(10000, 32,32,3)

#2.모델

mobileNetV2 = MobileNetV2(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))  

# mobileNetV2.trainable= False     

model = Sequential()
model.add(mobileNetV2)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100,activation='softmax'))

model.trainable = False

from sklearn.metrics import accuracy_score
model.compile(loss= 'sparse_categorical_crossentropy',optimizer='adam')
model.fit(x_train,y_train, epochs=10, batch_size=256, verbose=1)
model.evaluate(x_test,y_test)
y_predict =np.argmax(model.predict(x_test),axis=1)
acc = accuracy_score(y_test,y_predict)
print('acc : ', round(acc,4))


#mobileNetV2 False -- acc : acc :  0.011
#all True -- acc :  0.196
#all False -- acc :  acc :  0.125


