from matplotlib.pyplot import axis
from tensorflow.keras.datasets import fashion_mnist,mnist,cifar10
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

#1.데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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
randidx = np.random.randint(x_train.shape[0], size=augument_size)

x_augument = x_train[randidx].copy()
y_augument = y_train[randidx].copy()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
x_augument = x_augument.reshape(x_augument.shape[0], x_augument.shape[1], x_augument.shape[2], 3)

x_augumented = train_datagen.flow(x_augument, y_augument, batch_size=augument_size, shuffle=False).next()[0]

x_data_all = np.concatenate((x_train, x_augumented)) 

y_data_all = np.concatenate((y_train, y_augument))

xy_train = test_datagen.flow(x_data_all, y_data_all, batch_size=augument_size, shuffle=False)

xy_test = test_datagen.flow(x_test, y_test, batch_size=augument_size, shuffle=False)


np.save('d:/study_data/_save/_npy/cifar10/keras49_03_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/cifar10/keras49_03_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/cifar10/keras49_03_test_x.npy', arr=xy_test[0][0])
np.save('d:/study_data/_save/_npy/cifar10/keras49_03_test_y.npy', arr=xy_test[0][1])

