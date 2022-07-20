import numpy as np 
from keras.preprocessing.image import ImageDataGenerator

#1.데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,  #스케일링 자체 minmax제공
    horizontal_flip=True,#수평
    vertical_flip=True,#수직
    width_shift_range=0.1,#지정된 수평방향 이동
    height_shift_range=0.1,#수직방향 이동
    rotation_range=5,#각도 회전
    zoom_range=1.2,#확대
    shear_range=0.7,#찌그러 뜨리는거
    fill_mode = 'nearest') 
# 다 적용되지 않음. 랜덤으로 적용해서 변환함

test_datagen = ImageDataGenerator(
    rescale=1./255)    #test데이터는 증폭x 

xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/brain/train/', # 이 경로의 이미지파일을 불러 수치화
    target_size=(200,200),# 크기들을 일정하게 맞춰준다.
    batch_size=5,
    class_mode='binary', 
    color_mode='grayscale', #디폴트값은 컬러
    shuffle=True,
    )#Found 160 images belonging to 2 classes

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/image/brain/test/', # 이 경로의 이미지파일을 불러 수치화
    target_size=(200,200),# 크기들을 일정하게 맞춰준다.
    batch_size=5,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    )#Found 120 images belonging to 2 classes.

print(xy_train)
#<keras.preprocessing.image.DirectoryIterator object at 0x000002082E821D90>
print(xy_train[31]) #0~31 batch가 5이므로 160장이므로 32장으로 나뉘어짐
print(xy_train[31][0].shape) #x값
print(xy_train[31][1].shape) #y값
'''
print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))#<class 'tuple'> x,y값
print(type(xy_train[0][0]))#<class 'numpy.ndarray'>
print(type(xy_train[0][1]))#<class 'numpy.ndarray'>
# image 데이터를 가져왔을 때 x numpy y numpy 형태로 batch단위로 묶여있다!

# 현재 5,200,200,1 짜리 데이터가 32덩어리!

#2.모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape = (200,200,1),activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3.컴파일,훈련
model.compile(loss= 'binary_crossentropy',optimizer='adam',metrics= ['accuracy'])
# model.fit(xy_train[0][0],xy_train[0][1])# batch를 최대로 잡으면 이렇게도 가능.
from tensorflow.python.keras.callbacks import EarlyStopping
# es= EarlyStopping(monitor= 'val_loss',patience=20,mode='auto',restore_best_weights=True)
hist = model.fit_generator(xy_train, epochs= 40, steps_per_epoch=32,
                                        #전체데이터/batch = 160/5 = 32
                    validation_data=xy_test,
                    validation_steps=4) #val_steps: 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다. 

accuracy = hist.history['accuracy']
val_accuracy =  hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_accuracy : ', val_accuracy[-1])

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6)) #그래프 표 사이즈
plt.plot(hist.history['loss'], marker = '.' ,c = 'red', label = 'loss') # maker: 점으로 표시하겟다  ,c:색깔 ,label : 이름
plt.plot(hist.history['val_loss'], marker = '.' ,c = 'orange', label = 'val_loss') # maker: 점으로 표시하겟다  ,c:색깔 ,label : 이름
plt.plot(hist.history['accuracy'], marker = '.' ,c = 'black', label = 'accuracy') # maker: 점으로 표시하겟다  ,c:색깔 ,label : 이름
plt.plot(hist.history['val_accuracy'], marker = '.' ,c = 'blue', label = 'val_accuracy')
plt.grid() # 모눈종이에 하겠다
plt.title('gen')#제목
plt.ylabel('loss')#y축 이름
plt.xlabel('epochs')#x축 이름
plt.legend(loc='upper right') # upper right: 위쪽 상단에 표시하겠다.(라벨 이름들)
plt.show()# 보여줘

'''