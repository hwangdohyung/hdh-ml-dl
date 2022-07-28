from regex import D
import tensorflow as tf 
from tensorflow.python.keras.layers import Input, Dense, LeakyReLU,Reshape,Conv2D,Conv2DTranspose,Flatten,Dropout
from tensorflow.python.keras.models import Model
import numpy as np 
latent_dim = 32
height = 32 
width = 32 
channels= 3

generator_input = Input(shape= latent_dim)

x = Dense(128 *16 * 16)(generator_input)
x = LeakyReLU()(x)
x = Reshape((16, 16, 128))(x)

x = Conv2D(256, 5, padding = 'same')(x)
x = LeakyReLU()(x)

x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = LeakyReLU()(x)

x = Conv2D(256, 5, padding = 'same')(x)
x = LeakyReLU()(x)
x = Conv2D(256, 5, padding = 'same')(x)
x = LeakyReLU()(x)

x = Conv2D(channels, 7, activation='tanh',padding='same')(x)

generator = Model(generator_input, x)
generator.summary()

discriminator_input = Input(shape=(height,width,channels))

x = Conv2D(128,3)(discriminator_input)
x = LeakyReLU()(x)
x = Conv2D(128, 4, strides=2)(x)
x = LeakyReLU()(x)
x = Conv2D(128, 4, strides=2)(x)
x = LeakyReLU()(x)
x = Conv2D(128, 4, strides=2)(x)
x = LeakyReLU()(x)
x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(1, activation='sigmoid')(x)

discriminator = Model(discriminator_input,x)
discriminator.summary()

import tensorflow as tf 
from tensorflow.python.keras.optimizers import adam_v2


discriminator.compile(optimizer = 'adam', loss= 'binary_crossentropy')

discriminator.trainable = False 
gan_input = Input(shape=(latent_dim, ))
gan_output = discriminator(generator(gan_input)) 
gan = Model(gan_input, gan_output )

gan.compile(optimizer='adam',loss='binary_crossentropy')

import os 
from tensorflow.keras.datasets import cifar10 
from keras.preprocessing import image 
(x_train, y_train),(_,_) = cifar10.load_data()

x_train = x_train[y_train.flatten() == 6]
x_train = x_train.reshape((x_train.shape[0],)+(height,width,channels)).astype('float32')/255.

iterations = 10000
batch_size = 20

save_dir = 'D:\study_data\_data\image\pix2pix/'
if not os.path.exists(save_dir):
  os.mkdir(save_dir)

start = 0 
for step in range(iterations):
    random_latent_vectors = np.random.normal(size = (batch_size, latent_dim))
    generated_images = generator.predict(random_latent_vectors)
    
    stop = start +batch_size
    real_images = x_train[start:stop]
    combined_images = np.concatenate([generated_images, real_images])
    
    labels = np.concatenate([np.ones((batch_size,1)),
                            np.zeros((batch_size,1))])
    labels += 0.05 * np.random.random(labels.shape)
    
    d_loss = discriminator.train_on_batch(combined_images, labels)
    
    random_latent_vectors = np.random.normal(size=(batch_size,latent_dim))
    
    misleading_targets = np.zeros((batch_size,1))
    
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    
    start += batch_size
    if start > len(x_train) - batch_size:
      start = 0 
      
    if step % 100 == 0: 
      gan.save_weights('gan.h5')
    
      print('\nstep:{}'.format(step))
      print('discriminator loss: {}'.format(d_loss))
      print('adversarial loss: {}'.format(a_loss))
      
      
