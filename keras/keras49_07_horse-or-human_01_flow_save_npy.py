from matplotlib.pyplot import axis
from tensorflow.keras.datasets import fashion_mnist,mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator

#1.데이터
train_datagen0 = ImageDataGenerator(rescale=1./255,)  
test_datagen0 = ImageDataGenerator(rescale=1./255)   

xy = train_datagen0.flow_from_directory(
    'd:/study_data/_data/image/horse-or-human/', 
    target_size=(150,150),# 크기들을 일정하게 맞춰준다.
    batch_size=500,
    class_mode='binary', 
    # color_mode='grayscale', #디폴트값은 컬러
    shuffle=True,)

x=np.array(xy[0][0])
y=np.array(xy[0][1])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

augument_size = 64
randidx = np.random.randint(x.shape[0], size=augument_size)

x_augument = x[randidx].copy()
y_augument = y[randidx].copy()

x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 3)

x_augument = x_augument.reshape(x_augument.shape[0], x_augument.shape[1], x_augument.shape[2], 3)

x_augumented = train_datagen.flow(x_augument, y_augument, batch_size=augument_size, shuffle=False).next()[0]

x = np.concatenate((x, x_augumented)) 

y = np.concatenate((y, y_augument))

xy = test_datagen.flow(x, y, batch_size=augument_size, shuffle=False)


np.save('d:/study_data/_save/_npy/horse-or-human_02/keras49_07_x.npy', arr=xy[0][0])
np.save('d:/study_data/_save/_npy/horse-or-human_02/keras49_07_y.npy', arr=xy[0][1])


