import numpy as np 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255)

test_datagen = ImageDataGenerator(
    rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/cat_dog/training_set/training_set/', # 이 경로의 이미지파일을 불러 수치화
    target_size=(150,150),# 크기들을 일정하게 맞춰준다.
    batch_size=9000,
    class_mode='binary', 
    # color_mode='grayscale', #디폴트값은 컬러
    shuffle=True,
    )

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/image/cat_dog/test_set/test_set/', # 이 경로의 이미지파일을 불러 수치화
    target_size=(150,150),# 크기들을 일정하게 맞춰준다.
    batch_size=9000,
    class_mode='binary',
    # color_mode='grayscale',
    shuffle=True,
    )

# print(xy_train)
print(xy_test)

np.save('d:/study_data/_save/_npy/cat_dog/keras47_01_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/cat_dog/keras47_01_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/cat_dog/keras47_01_test_x.npy', arr=xy_test[0][0])
np.save('d:/study_data/_save/_npy/cat_dog/keras47_01_test_y.npy', arr=xy_test[0][1])

