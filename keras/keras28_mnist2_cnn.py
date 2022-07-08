from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D,Flatten,MaxPool2D # 이미지 작업은 2차원
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
import pandas as pd
import numpy as np

#1.데이터
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(np.unique(y_train, return_counts = True)) # 10개의 다중분류 softmax , loss 카테고리컬 , onehot 인코딩
print(y_train.shape,y_test.shape)




'''
# # one hot encoding
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y_train) 
# # 트레인 테스트 분리
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=True,
#                                                      random_state=58)

#2.모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3),padding ='same',input_shape = (28, 28, 1)))    
model.add(MaxPool2D())
model.add(Conv2D(32, (2,2),padding='valid',activation ='relu'))
model.add(MaxPool2D())  
model.add(Flatten()) 
model.add(Dense(32,activation = 'relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))
model.summary() 

#만들기! acc 0.98이상

'''