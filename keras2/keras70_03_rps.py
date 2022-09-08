import numpy as np 
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,Input
from keras.applications import VGG19,VGG16,Xception,ResNet50,ResNet101,InceptionResNetV2,InceptionV3,DenseNet121,MobileNetV2,EfficientNetB0
from keras.datasets import cifar10,cifar100
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
tf.random.set_seed(123)
import warnings
warnings.filterwarnings('ignore')


x = np.load('d:/study_data/_save/_npy/rps_02/keras49_08_x.npy')
y = np.load('d:/study_data/_save/_npy/rps_02/keras49_08_y.npy')

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8,shuffle=True,random_state=66,stratify=y)

x_train = x_train.reshape(51, 150*150*3)
x_test = x_test.reshape(13, 150*150*3)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(51, 150,150,3)
x_test = x_test.reshape(13, 150,150,3)

# from keras.utils import to_categorical
# y_trian = to_categorical(y_train)
# y_test = to_categorical(y_test)



vGG16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(150, 150, 3))  

# vGG16.trainable= False     

model = Sequential()
model.add(vGG16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(3,activation='softmax'))

# model.trainable = False

from sklearn.metrics import accuracy_score
model.compile(loss= 'categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train, epochs=100, batch_size=256, verbose=1)
model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1) 
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_predict)
print('acc : ', round(acc,4))


#vgg False - acc :  0.7692
#all True - acc :  0.4615


