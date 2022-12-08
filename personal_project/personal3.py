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
down_result = down_model(tf.expand_dims(a128, 0))
print (down_result.shape)

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

up_model = upsample(3,4)
up_result = up_model(down_result)

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

generator = Generator()

gen_output = generator(a128[tf.newaxis,...],training=False)

plt.imshow(gen_output[0,...])
 

############################### DISCRIMINATOR ##################################
 
def Discriminator():
    initializer = tf.random_normal_initializer(0.,0.02)
    
    inp = tf.keras.layers.Input(shape =[128,128,3], name = 'input_image')
    tar = tf.keras.layers.Input(shape =[128,128,3], name = 'target_image')
    
    x = tf.keras.layers.concatenate([inp,tar]) 
    
    down1 = downsample(64, 4, False)(x) 
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer =initializer,
                                  use_bias=False)(zero_pad1)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    
    last = tf.keras.layers.Conv2D(1, 4, strides=1, 
                                  kernel_initializer=initializer)(zero_pad2)    
    
    return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()

################################# loss ###################################
LAMBDA =100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output,disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss

def generator_loss(disc_generated_output,gen_output,target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output),disc_generated_output)
    
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    total_gen_loss = gan_loss + (LAMBDA*l1_loss)
    
    return total_gen_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4,beta_1=-0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4,beta_1=-0.5)

############################## checkpoint ################################

checkpoint_dir = 'D:\study_data\_temp'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

############################### train #####################################
EPOCHS = 150 

def generate_images(model, test_input, tar): 
    prediction = model(test_input, training =True)
    plt.figure(figsize=(15,15))
    
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    for i in range(3):
        plt.subplot(1, 3, 1+i)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 +0.5)
        plt.axis('off')
    plt.show()
    
    
@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        
        disc_real_output = discriminator([input_image])
        disc_generated_output = discriminator([input_image, gen_output], training = True)
        
        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    
    generator_gradients = gen_tape.gradient(gen_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))
    
import time
def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time() 
        
        for input_image,target in train_ds:
            train_step(input_image, target)
        
        for example_input, example_target in test_ds.take(1):
            generate_images(generator, example_input, example_target)       
        
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            
        
        print('Time taken for epoch {} is {} sec/n'.format(epoch + 1,
                                                           time.time()-start))
        
        
fit(train_dataset, EPOCHS, test_dataset)    



