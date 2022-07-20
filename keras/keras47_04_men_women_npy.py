import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from regex import escape
 
x = np.load('D:\study_data\_save\_npy/men_women/keras47_04_x.npy.')
y= np.load('D:\study_data\_save\_npy/men_women/keras47_04_y.npy')


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8,shuffle=True,random_state=66)


#2.모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape = (150,150,3),activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3.컴파일,훈련
model.compile(loss= 'binary_crossentropy',optimizer='adam')
# batch를 최대로 잡으면 이렇게도 가능.
from tensorflow.python.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor= 'val_loss',patience=20,mode='auto',restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=1,validation_split=0.1,verbose=1,batch_size=32,callbacks=es)
# hist = model.fit_generator(x_train,y_train, epochs= 40, steps_per_epoch=32,
#                                         #전체데이터/batch = 160/5 = 32
#                     validation_data=x_train,
#                     validation_steps=4) #val_steps: 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다. 

# accuracy = hist.history['accuracy']
# val_accuracy =  hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# print('loss : ', loss[-1])
# print('val_loss : ', val_loss[-1])
# print('accuracy : ', accuracy[-1])
# print('val_accuracy : ', val_accuracy[-1])

yys = ImageDataGenerator(
    rescale=1./255)

yys = yys.flow_from_directory(
    'D:\study_data\_data\image\dddd', # 이 경로의 이미지파일을 불러 수치화
    target_size=(150,150),# 크기들을 일정하게 맞춰준다.
    batch_size=9000,
    class_mode='binary', 
    # color_mode='grayscale', #디폴트값은 컬러
    shuffle=True,
    )
print(yys.shape)
'''

#4.평가,예측 
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(yys)

from sklearn.metrics import accuracy_score
y_predict = y_predict.round()
acc = accuracy_score(y_test,y_predict)

print('loss : ', loss)
print('acc : ', acc)

print(y_predict)

'''