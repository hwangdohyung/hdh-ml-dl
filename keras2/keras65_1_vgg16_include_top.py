import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.applications import VGG16

# model = VGG16() # include_top=True, input_shape(224, 224, 3)
model = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))  #defalut는 True

model.summary()

print(len(model.weights))           #32 (weight , bias)
print(len(model.trainable_weights)) #32

########################### include_top = True #######################
#1. FC layer 원래꺼 그대로 쓴다. 
#2. input_shape= (224, 224, 3) 고정값 - 바꿀 수 없다.

# input_1 (InputLayer)        [(None, 224, 224, 3)]     0

# block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792

# block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928
# ...........................................

# flatten (Flatten)           (None, 25088)             0

# fc1 (Dense)                 (None, 4096)              102764544

# fc2 (Dense)                 (None, 4096)              16781312

# predictions (Dense)         (None, 1000)              4097000

########################### include_top = False ##########################
#1. FC layer(flatten 부터)사라짐.
#2. input_shape=(32, 32, 3) - 내가 설정한 shape 

#  input_1 (InputLayer)        [(None, 32, 32, 3)]       0

#  block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792

#  block1_conv2 (Conv2D)       (None, 32, 32, 64)        36928

# #.............................................................

#  block5_conv1 (Conv2D)       (None, 2, 2, 512)         2359808

#  block5_conv2 (Conv2D)       (None, 2, 2, 512)         2359808

#  block5_conv3 (Conv2D)       (None, 2, 2, 512)         2359808

#  block5_pool (MaxPooling2D)  (None, 1, 1, 512)         0



