import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
 
x = np.load('d:/study_data/_save/_npy/rps_02/keras49_08_x.npy')
y = np.load('d:/study_data/_save/_npy/rps_02/keras49_08_y.npy')

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.7,shuffle=True,random_state=66)


#2.모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten

model = Sequential()
model.add(Conv2D(128, (2,2), input_shape = (150,150,3),activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor= 'val_loss',patience=30,mode='auto',restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=300,validation_split=0.1,verbose=1,batch_size=32,callbacks=es)

#4.평가,예측 
loss = hist.history['loss']
accuracy = hist.history['accuracy']

print('loss: ', loss[-1])
print('accuracy: ', accuracy[-1])


# loss:  1.0750298500061035
# accuracy:  0.43589743971824646