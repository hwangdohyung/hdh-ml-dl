import numpy as np 
from keras.datasets import mnist 
from sklearn.decomposition import PCA
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

(x_train, y_train),(x_tset, y_test) = mnist.load_data()

for
