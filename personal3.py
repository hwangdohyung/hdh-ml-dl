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

LAMBDA = 100
BATCH_SIZE = 16
BUFFER_SIZE  = 400

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE)
test_dataset = test_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE)

########################## 모델 #######################

OUTPUT_CHANNELS = 3

##### 다운샘플 정의 #####
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.,0.02)
    
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    
    result.add(tf.keras.layers.LeakyReLU())
    
    return result 

down_model = downsample(3,4)


###### 업샘플 정의 #####
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer = initializer,
                                        use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
        
    result.add(tf.keras.layers.ReLU())
    
    return result

###################### 제너레이터(업샘플+다운샘플)u_net ####################
def Generator():
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),]
    
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4,),
        upsample(256, 4,),
        upsample(128, 4,),
        upsample(64, 4,)]
    
    initializer = tf.random_normal_initializer(0.,0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')
    
    concat = tf.keras.layers.Concatenate()
    
    inputs = tf.keras.layers.Input(shape=[None,None,3])
    x = inputs
    
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
     
    skips = reversed(skips[:-1])
    
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x,skip])    
    x = last(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

############# 이미지 1장 빼놓은거 ##########
img  = cv2.imread('D:\study_data\_data\image\gan\color/image0000.jpg')
img  = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (128,128))
a128 = img_to_array(Image.fromarray(img))

a128 = (a128/127.5) - 1

generator = Generator()

gen_output = generator(a128[tf.newaxis,...],trainig=False)
plt.imshow(gen_output[0,...])
    
    
    
    
    
    
    
    
    