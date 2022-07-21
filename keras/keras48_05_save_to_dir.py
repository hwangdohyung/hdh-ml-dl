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
                                  save_to_dir = 'D:\study_data\_temp',
                                  shuffle=False).next()[0] # 셔플은 false 이미 섞여있으므로

print(x_augmented)
print(x_augmented.shape)# (40000,28,28,1) #변환된 놈

################################## 기본 데이터 6만장 + 증폭데이터 4만장 ##################################
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))   # 괄호 2개의미 찾기 클래스*

print(x_train.shape,y_train.shape)  #(100000,28,28,1) (100000, )


