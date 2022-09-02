from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip= True,
    vertical_flip= True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augment_size = 100

print(x_train[0].shape)                 #(28,28)
print(x_train[0].reshape(28*28).shape)  #(784,)
print(np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1).shape) #(100,28,28,1)
#np.tile:증폭시켜줌, 변환은 안됨.
# [5, 6] 형태의 2개의 원소를 가진 배열을 선언하고,

# 가로로만 3차원, 그리고 (2, 2) 차원으로 쌓기를 해보겠습니다.

# import numpy as np

# a = np.array([5, 6])

# np.tile(a, 3)
# # array([5, 6, 5, 6, 5, 6]), shape = (6,)

# np.tile(a, (2, 2))
# '''array([[5, 6, 5, 6],
#        [5, 6, 5, 6]]) shape = (2, 4)
########################################################################
print(np.zeros(augment_size))
print(np.zeros(augment_size).shape)   #(100,)


x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),   # x값
    np.zeros(augment_size),                                                    # y값
    batch_size=augment_size,
    shuffle = True,
    
).next()


#############################넥스트 사용###############################
# print(x_data)
# print(x_data[0])
# print(x_data[0].shape) 
# print(x_data[1].shape)

#############################넥스트 미사용###############################
print(x_data)
#<keras.preprocessing.image.NumpyArrayIterator object at 0x000001E5A0CB58B0>
print(x_data[0])           # x와 y가 모두 포함 ,batch단위로 묶여있는것 
print(x_data[0][0].shape)  #(100,28,28,1) 
print(x_data[0][1].shape)  #(100, )

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7)) 
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i],cmap = 'gray') # .next()사용
    #plt.imshow(x_data[0][0][i], cmap='gray')    # .next()미사용
plt.show()

# .next에 대해서 알아보기 *과제! iterate도
# 분류모델 할때 데이터 갯수를 증폭으로 맞춰주면 성능 개선 가능성이 있다.

