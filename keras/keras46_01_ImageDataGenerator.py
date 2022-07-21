import numpy as np 
from keras.preprocessing.image import ImageDataGenerator

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
    target_size=(150,150),# 크기들을 일정하게 맞춰준다.
    batch_size=5,
    class_mode='binary', 
    # color_mode='grayscale', #디폴트값은 컬러
    shuffle=True,
    )#Found 160 images belonging to 2 classes

xy_test = test_datagen.flow_from_directory(
   'd:/study_data/_data/image/brain/test/', # 이 경로의 이미지파일을 불러 수치화
    target_size=(150,150),# 크기들을 일정하게 맞춰준다.
    batch_size=5,
    class_mode='binary',
    # color_mode='grayscale',
    shuffle=True,
    )#Found 120 images belonging to 2 classes.

print(xy_train)

#<keras.preprocessing.image.DirectoryIterator object at 0x000002082E821D90>
print(xy_train[0]) #0~31 batch가 5이므로 160장이므로 31장으로 나뉘어짐
print(xy_train[0][0].shape) #x값
print(xy_train[0][1].shape) #y값

print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))#<class 'tuple'> x,y값
print(type(xy_train[0][0]))#<class 'numpy.ndarray'>
print(type(xy_train[0][1]))#<class 'numpy.ndarray'>
# image 데이터를 가져왔을 때 x numpy y numpy 형태로 batch단위로 묶여있다!

