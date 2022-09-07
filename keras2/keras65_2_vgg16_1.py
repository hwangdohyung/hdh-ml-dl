import numpy as np 
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.applications import VGG16,ResNet152V2
from keras.datasets import cifar10

vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))  
vgg16.trainable=False       # vgg16 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10,activation='softmax'))

# model.trainable = True

model.summary()

                                        # Trainable:True  # VGG False   # model False
print(len(model.weights))               # 30 / 30 / 30
print(len(model.trainable_weights))     # 30 /  4 /  0




