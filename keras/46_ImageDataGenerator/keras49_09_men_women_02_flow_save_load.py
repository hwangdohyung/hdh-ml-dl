from matplotlib.pyplot import hist2d
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# 1. 데이터
x_train = np.load('d:/study_data/_save/_npy/men_women_02/keras49_09_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/men_women_02/keras49_09_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/men_women_02/keras49_09_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/men_women_02/keras49_09_test_y.npy')

# mypic = np.load('d:/study_data/_save/_npy/keras47_04_mypic.npy')

print(x_train.shape) # (2016, 150, 150, 3)
print(y_train.shape) # (2016,)
print(x_test.shape) # (504, 150, 150, 3)
print(y_test.shape) # (504,)

# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(150,150,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

hist = model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.2) 

# 그래프
loss = hist.history['loss']
accuracy = hist.history['accuracy']
val_loss = hist.history['val_loss']
val_accuracy = hist.history['val_accuracy']


print('loss: ', loss[-1])
print('accuracy: ', accuracy[-1])
print('val_loss: ', val_loss[-1])
print('val_accuracy: ', val_accuracy[-1])

# loss:  0.005300433840602636
# accuracy:  0.9992054104804993
