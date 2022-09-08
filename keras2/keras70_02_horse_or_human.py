import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,Input
from keras.applications import VGG19,VGG16
from keras.datasets import cifar10,cifar100
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
tf.random.set_seed(123)

x = np.load('d:/study_data/_save/_npy/horse-or-human_02/keras49_07_x.npy')
y= np.load('d:/study_data/_save/_npy/horse-or-human_02/keras49_07_y.npy')

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8,shuffle=True,random_state=66)


x_train = x_train.reshape(51, 150*150*3)
x_test = x_test.reshape(13, 150*150*3)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(51, 150,150,3)
x_test = x_test.reshape(13, 150,150,3)

#2.모델
input1 = Input(shape=(150,150,3))
vgg16 = VGG16(weights='imagenet',include_top=False)(input1)
gap1 = Flatten()(vgg16)
# gap1 = GlobalAveragePooling2D()(vgg16)    
hidden1 = Dense(100)(gap1)
output = Dense(1,activation='sigmoid')(hidden1)
model = Model(inputs=input1,outputs=output)

# model.trainable = False
model.layers[1].trainable= False
model.summary()


from sklearn.metrics import accuracy_score
model.compile(loss= 'binary_crossentropy',optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=256, verbose=1)
model.evaluate(x_test,y_test)
y_predict = (model.predict(x_test)).round()
acc = accuracy_score(y_test,y_predict)
print('acc : ', round(acc,4))

# acc :  1


