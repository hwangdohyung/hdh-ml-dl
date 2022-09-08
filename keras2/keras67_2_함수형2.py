import numpy as np 
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,Input
from keras.applications import VGG16,ResNet152V2,InceptionV3,DenseNet201
from keras.datasets import cifar10,cifar100
import tensorflow as tf 

(x_train,y_train),(x_test,y_test)  = cifar100.load_data()

x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)

base_model = DenseNet201(weights='imagenet',include_top=False,input_shape=(32,32,3))
base_model.summary()


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(102, activation='relu')(x)

output1 = Dense(100,activation='softmax')(x)
model = Model(inputs=base_model.input, outputs= output1)

for layer in base_model.layers:        # base_model.layers[3]  3번째 레이어만 동결
    layer.trainable = False

# base_model.trainable = False 
   
model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
model.fit(x_train,y_train,epochs=10,verbose=1)

model.evaluate(x_test,y_test)
y_predict= np.argmax(model.predict(x_test),axis=1)

from sklearn.metrics import accuracy_score
print('acc : ',accuracy_score(y_predict,y_test))



