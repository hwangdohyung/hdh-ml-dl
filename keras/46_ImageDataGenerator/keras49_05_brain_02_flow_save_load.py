import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from warnings import filters
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical 
from sklearn.preprocessing import OneHotEncoder  
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
x_train = np.load('d:/study_data/_save/_npy/brain/keras49_05_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/brain/keras49_05_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/brain/keras49_05_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/brain/keras49_05_test_y.npy')

 # 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
#2. 모델구성
input1 = Input(shape=(150,150,1))
conv2D_1 = Conv2D(100,3, padding='same')(input1)
MaxP1 = MaxPooling2D()(conv2D_1)
drp1 = Dropout(0.2)(MaxP1)
conv2D_2 = Conv2D(200,2,
                  activation='relu')(drp1)
MaxP2 = MaxPooling2D()(conv2D_2)
drp2 = Dropout(0.2)(MaxP2)
conv2D_3 = Conv2D(200,2, padding='same',
                  activation='relu')(drp2)
MaxP3 = MaxPooling2D()(conv2D_3)
drp3 = Dropout(0.2)(MaxP3)
flatten = Flatten()(drp3)
dense1 = Dense(200)(flatten)
batchnorm1 = BatchNormalization()(dense1)
activ1 = Activation('relu')(batchnorm1)
drp4 = Dropout(0.2)(activ1)
dense2 = Dense(100)(drp4)
batchnorm2 = BatchNormalization()(dense2)
activ2 = Activation('relu')(batchnorm2)
drp5 = Dropout(0.2)(activ2)
dense3 = Dense(100)(drp5)
batchnorm3 = BatchNormalization()(dense3)
activ3 = Activation('relu')(batchnorm3)
drp6 = Dropout(0.2)(activ3)
output1 = Dense(1, activation='sigmoid')(drp6)
model = Model(inputs=input1, outputs=output1)   

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True,mode = 'auto',verbose=1)
hist = model.fit(x_train,y_train, epochs= 200, validation_split=0.1, verbose=1,callbacks=es)

# 4. 평가, 예측
loss = hist.history['loss']
accuracy = hist.history['accuracy']

print('loss: ', loss[-1])
print('accuracy: ', accuracy[-1])

# loss:  0.693456768989563
# accuracy:  0.5263158082962036