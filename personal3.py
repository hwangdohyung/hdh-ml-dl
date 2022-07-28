import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

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

X = []
y = []

for i in range(5000) :
    
    img1 = cv2.cvtColor(cv2.imread(clr_img_path[i]), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(gry_img_path[i]), cv2.COLOR_BGR2RGB)
    
    y.append(img_to_array(Image.fromarray(cv2.resize(img1,(128,128)))))
    X.append(img_to_array(Image.fromarray(cv2.resize(img2,(128,128)))))

X = np.array(X)
y = np.array(y)

inp = (gry_img_path +'image1000.jpg')    
# plt.figure(figsize = (10,50))

# i = 0

# while i < 20:
    
#     x = np.random.randint(0,3000)
    
#     plt.subplot(10, 2, i+1)
#     plt.imshow(X[x]/255.0,'gray')
#     plt.axis('off')
#     plt.title('Gray Image')
    
#     plt.subplot(10, 2, i+2)
#     plt.imshow(y[x]/ 255.0)
#     plt.axis('off')
#     plt.title('ColorImage')
#     i += 2
    
# plt.show()

X = (X/127.5) - 1
y = (y/127.5) - 1

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle = False)

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

LAMBDA = 100
BATCH_SIZE = 16
BUFFER_SIZE  = 400

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE)
test_dataset = test_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE)

OUTPUT_CHANNERLS = 3

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

def upsample(filters, size, apply_dorpout=False):
    initializer = tf.random_normal_initializer(0.,0.02)
    
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2, 
                                        padding= 'same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    
    result.add(tf.keras.layers.BatchNormalization())
    
    if apply_dorpout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    
    return result
def Generator():
    inputs = tf.keras.layers.Input(shape=[256,256,3])
    
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]     
    
    up_stack = [
        upsample(512, 4 , apply_dorpout=True),
        upsample(512, 4 , apply_dorpout=True),
        upsample(512, 4 , apply_dorpout=True),
        upsample(512, 4 ),
        upsample(256, 4 ),
        upsample(128, 4 ),
        upsample(64, 4 ),
    ]
    
    initializer = tf.random_normal_initializer(0.,0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNERLS, 4,
                                           strides=2,
                                           padding= 'same',
                                           kernel_initializer=initializer,
                                           activation='tanh')
    
    x= inputs
    
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x= up(x)
        x= tf.keras.layers.Concatenate()([x,skip])
        
    x = last(x)
    
    return tf.keras.Model(input=inputs, outputs=x)
generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True,dpi=64)

LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
     
  # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
 
    return total_gen_loss, gan_loss, l1_loss

