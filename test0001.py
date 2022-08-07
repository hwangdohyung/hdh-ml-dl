import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow_addons.layers import SpectralNormalization
from keras.layers import BatchNormalization
from keras.layers import ZeroPadding2D
from keras.layers import Concatenate
from keras.layers import Activation
from keras.models import Model
from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Conv2DTranspose
from keras.initializers import RandomNormal
from tensorflow_addons.layers import InstanceNormalization

#################### 데이터 ####################
clr_path = "D:\study_data\_data\image\gan\color"
gry_path = "D:\study_data\_data\image\gan\gray"

import os

clr_img_path = []
gry_img_path = []

for img_path in os.listdir(clr_path) :
    clr_img_path.append(os.path.join(clr_path, img_path))
    
for img_path in os.listdir(gry_path) :
    gry_img_path.append(os.path.join(gry_path, img_path))

clr_img_path.sort()
gry_img_path.sort()

from PIL import Image
from keras.preprocessing.image import img_to_array

x = []
y = []

for i in range(5000) :
    
    img1 = cv2.cvtColor(cv2.imread(clr_img_path[i]), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(gry_img_path[i]), cv2.COLOR_BGR2RGB)
    
    y.append(img_to_array(Image.fromarray(cv2.resize(img1,(128,128)))))
    x.append(img_to_array(Image.fromarray(cv2.resize(img2,(128,128)))))

x = np.array(x)
y = np.array(y)
    
x = (x/127.5) - 1
y = (y/127.5) - 1

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.15, shuffle = False)
######################### 1장################################
img  = cv2.imread('D:\study_data\_data\image\gan\color/image0000.jpg')
img  = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (256,256))
a128 = img_to_array(Image.fromarray(img))
a128 = np.array(a128)
a128 = (a128/127.5) - 1
# a128 = a128.reshape(1,128,128,3)
##############################################################

LAMBDA = 100
BATCH_SIZE = 16
BUFFER_SIZE  = 400

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE)
test_dataset = test_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE)


