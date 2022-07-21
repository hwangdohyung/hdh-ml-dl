import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
 
x_train = np.load('d:/study_data/_save/_npy/cifar10/keras49_03_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/cifar10/keras49_03_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/cifar10/keras49_03_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/cifar10/keras49_03_test_y.npy')

 # 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(32,32,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True,mode = 'auto',verbose=1)
hist = model.fit(x_train,y_train, epochs= 200, validation_split=0.1, verbose=1,callbacks=es)

# 4. 평가, 예측
loss = hist.history['loss']
accuracy = hist.history['accuracy']

print('loss: ', loss[-1])
print('accuracy: ', accuracy[-1])

# loss:  0.003395141800865531
# accuracy:  1.0