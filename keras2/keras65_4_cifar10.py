import numpy as np 
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.applications import VGG16,ResNet152V2
from keras.datasets import cifar10
# 1.데이터
(x_train,y_train),(x_test,y_test)  = cifar10.load_data()

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32,32,3)
x_test = x_test.reshape(10000, 32,32,3)


#2.모델

vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))  

# vgg16.trainable= False     # vgg16 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10,activation='softmax'))

# model.trainable = False

from sklearn.metrics import accuracy_score
model.compile(loss= 'sparse_categorical_crossentropy',optimizer='adam')
model.fit(x_train,y_train, epochs=10, batch_size=256, verbose=1)
model.evaluate(x_test,y_test)
y_predict =np.argmax(model.predict(x_test),axis=1)
acc = accuracy_score(y_test,y_predict)
print('acc : ', round(acc,4))

#vgg False -- 0.5852
#all True --  0.8135
#all False -- 0.0927



