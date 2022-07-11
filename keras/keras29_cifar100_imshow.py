import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


import matplotlib.pyplot as plt
plt.imshow(x_train[20],)
plt.show()

