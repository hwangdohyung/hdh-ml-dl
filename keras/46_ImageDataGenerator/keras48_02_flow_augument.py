from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape)

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip= True,
    # vertical_flip= True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size= augment_size) #0~59999 번째 중 임의의 것으로 4만장 뽑음.
print(x_train.shape[0]) # 60000
print(randidx)          # [53513  8719 49228 ... 16351 51493 51918]
print(np.min(randidx),np.max(randidx))      
print(type(randidx))

# x_augment = x_train[randidx]# 카피는 안해도 됨. 저장하려면 copy
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape) # (40000, 28, 28)
print(y_augmented.shape) # (40000,)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1) # 위 모양과 같은 모양 ㅎㅎ

################################## 변환 작업 ###################################
x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                    x_augmented.shape[1],
                                    x_augmented.shape[2],1)


x_augmented = train_datagen.flow(x_augmented, y_augmented,
                                  batch_size=augment_size,
                                  shuffle=False).next()[0] # 셔플은 false 이미 섞여있으므로

print(x_augmented)
print(x_augmented.shape)# (40000,28,28,1) #변환된 놈

################################## 기본 데이터 6만장 + 증폭데이터 4만장 ##################################
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))   # 괄호 2개의미 찾기 클래스*

print(x_train.shape,y_train.shape)  #(100000,28,28,1) (100000, )


'''
print(x_train[0].shape)                 #(28,28)
print(x_train[0].reshape(28*28).shape)  #(784,)
print(np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1).shape) #(100,28,28,1)

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


############################### rand int ################################
#random.randint()
# random.randint() 함수는 [최소값, 최대값)의 범위에서 임의의 정수를 만듭니다.

# 예제1 - 기본 사용

# a = np.random.randint(2, size=5)
# print(a)

# b = np.random.randint(2, 4, size=5)
# print(b)

# c = np.random.randint(1, 5, size=(2, 3))
# print(c)
# [0 0 0 0 0]

# [3 3 2 2 3]

# [[3 2 4]
#  [2 2 2]]

# np.random.randint(2, size=5)는 [0, 2) 범위에서 다섯개의 임의의 정수를 생성합니다.
# np.random.randint(2, 4, size=5)는 [2, 4) 범위에서 다섯개의 임의의 정수를 생성합니다.
# np.random.randint(1, 5, size=(2, 3))는 [1, 5) 범위에서 (2, 3) 형태의 어레이를 생성합니다.

'''
