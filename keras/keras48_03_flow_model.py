from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)
print(x_test.shape)

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

# x_augument = x_train[randidx]# 카피는 안해도 됨. 저장하려면 copy
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
                                  shuffle=False).next()[0] # [0]:x값만 셔플은 false 이미 섞여있으므로

print(x_augmented)
print(x_augmented.shape)# (40000,28,28,1) #변환된 놈

################################## 기본 데이터 6만장 + 증폭데이터 4만장 ##################################
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape,y_train.shape)  #(100000,28,28,1) (100000, )


x_train = x_train.reshape(100000,28*28*1)
x_test = x_test.reshape(10000,28*28*1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2.모델구성
input1= Input(shape=(28*28*1))
dense1 = Dense(128,activation='relu')(input1)
dense2 = Dense(64,activation='relu')(dense1)
dense3 = Dense(32,activation='relu')(dense2)
output1= Dense(10,activation='softmax')(dense3)
model= Model(inputs=input1,outputs=output1)

#3.컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor= 'val_loss',patience=10, mode= 'auto', restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=20,validation_split=0.1,callbacks=es)

# print(y_test)

#4.평가 훈련
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
# print(y_test)
# print(y_predict)
y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print(':acc스코어 ', acc)

# :acc스코어  0.858
