import numpy as np 
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.applications import VGG16,ResNet152V2
from keras.datasets import cifar10

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))  

# vgg16.trainable=False       # vgg16 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10,activation='softmax'))

###################################### 2번 소스에서 아래만 추가 ###################################

print(model.layers)

import pandas as pd
pd.set_option('max_colwidth', -1)
layers= [(layer, layer.name, layer.trainable) for layer in model.layers]
resluts = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(resluts)