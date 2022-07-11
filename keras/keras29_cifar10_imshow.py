import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)


import matplotlib.pyplot as plt
plt.imshow(x_train[5],'gray')
plt.show()

