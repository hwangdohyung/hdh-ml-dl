from sklearn.model_selection import train_test_split
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,Input
from keras.applications import VGG19,VGG16,Xception,ResNet50,ResNet101,InceptionResNetV2,InceptionV3,DenseNet121,MobileNetV2,EfficientNetB0
from keras.datasets import cifar10,cifar100
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
tf.random.set_seed(123)
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x_train = np.load('d:/study_data/_save/_npy/men_women_02/keras49_09_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/men_women_02/keras49_09_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/men_women_02/keras49_09_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/men_women_02/keras49_09_test_y.npy')

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

x_train = x_train.reshape(3147, 150*150*3)
x_test = x_test.reshape(662, 150*150*3)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(3147, 150,150,3)
x_test = x_test.reshape(662, 150,150,3)

m_list=[VGG19,VGG16,Xception,ResNet50,ResNet101,InceptionResNetV2,InceptionV3,DenseNet121,MobileNetV2,EfficientNetB0]
acc_list=[]
for i in m_list:
#2.모델
    input1 = Input(shape=(150,150,3))
    vgg16 = VGG16(weights='imagenet',include_top=False)(input1)
    # gap1 = Flatten()(vgg16)
    gap1 = GlobalAveragePooling2D()(vgg16)    
    hidden1 = Dense(100)(gap1)
    output = Dense(1,activation='sigmoid')(hidden1)
    model = Model(inputs=input1,outputs=output)

    # model.trainable = False
    model.layers[1].trainable= False
    model.summary()

    from sklearn.metrics import accuracy_score
    model.compile(loss= 'binary_crossentropy',optimizer='adam',metrics=['acc'])
    model.fit(x_train,y_train, epochs=100, batch_size=256, verbose=0)
    model.evaluate(x_test,y_test)
    y_predict = (model.predict(x_test)).round()
    acc = accuracy_score(y_test,y_predict)
    print('acc : ', round(acc,4))
    acc_list.append([i.__name__,acc])
print(acc_list)







