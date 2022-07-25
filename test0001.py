import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Dropout,LSTM,Input,LeakyReLU,Reshape,Conv2DTranspose,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping

latent_dim = 32 
height = 32 
width = 32 
channels = 3

generator_input = Input(shape =latent_dim, )
x = Dense(128 *16* 16)(generator_input)
x = LeakyReLU()(x)
x = Reshape((16,16,128))(x)

x = Conv2D(256, 5, padding = 'same')(x)
x = LeakyReLU()(x)

x = Conv2DTranspose(256, 4, strides= 2, padding = 'same')(x)
x = LeakyReLU()(x)

x = Conv2D(256, 5, padding = 'same')(x)
x = LeakyReLU()(x)
x = Conv2D(256, 5, padding = 'same')(x)
x = LeakyReLU()(x)

x = Conv2D(channels, 7, activation= 'tanh', padding = 'same')(x)

generator  = Model(generator_input, x)
generator.summary()

discriminator_input = Input(shape=(height,width,channels))
x1 = Conv2D(128,3)(discriminator_input)
x1 = LeakyReLU()(x1)
x1 = Conv2D(128,4,strides=2)(x1)
x1 = LeakyReLU()(x1)
x1 = Conv2D(128,4,strides=2)(x1)
x1 = LeakyReLU()(x1)
x1 = Conv2D(128,4,strides=2)(x1)
x1 = LeakyReLU()(x1)
x1 = Flatten()(x1)
x1 = Dropout(0.4)(x1)
x1 = Dense(1, activation='sigmoid')(x1)

discriminator = Model(discriminator_input, x1)
discriminator.summary()

from tensorflow.python.keras.optimizers import rmsprop_v2
discriminator_optimizer = rmsprop_v2(lr= 0.0008, clipvalue = 1.0, decay =1e-8)
discriminator.compile(optimizer= discriminator_optimizer, loss= 'binary_crossentropy')
