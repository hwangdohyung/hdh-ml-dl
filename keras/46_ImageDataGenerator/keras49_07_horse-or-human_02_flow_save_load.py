import numpy as np 
from keras.preprocessing.image import ImageDataGenerator


x = np.load('d:/study_data/_save/_npy/horse-or-human_02/keras49_07_x.npy')
y= np.load('d:/study_data/_save/_npy/horse-or-human_02/keras49_07_y.npy')


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
model.add(Dense(1, activation='sigmoid'))


#3.컴파일,훈련
model.compile(loss= 'binary_crossentropy',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor= 'val_loss',patience=20,mode='auto',restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=300,validation_split=0.1,verbose=1,batch_size=32,callbacks=es)

#4.평가,예측 
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score
y_predict = y_predict.round()
acc = accuracy_score(y_test,y_predict)

print('loss : ', loss)
print('acc : ', acc)

# loss :  0.6701513528823853
# acc :  0.7