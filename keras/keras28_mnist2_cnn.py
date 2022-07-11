from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D,Flatten,MaxPool2D # 이미지 작업은 2차원
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,RobustScaler,StandardScaler

#1.데이터
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(np.unique(y_train, return_counts = True)) # 10개의 다중분류 softmax , loss 카테고리컬 , onehot 인코딩
print(y_train.shape,y_test.shape)



# one hot encoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y_train) 
y = to_categorical(y_test)

# minmax , standard
# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)#스케일링한것을 보여준다.
# x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 

#2.모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3),padding ='same',input_shape = (28, 28, 1)))    
model.add(MaxPool2D())
model.add(Conv2D(32, (2,2),padding='valid',activation ='relu'))
model.add(MaxPool2D())  
model.add(Flatten()) 
model.add(Dense(32,activation = 'relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))

#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer ='adam', metrics='accuracy') #다중분류는 무조건 loss에 categorical_crossentropy
 #분류모델에서 셔플 중요! ,false로 하면 순차적으로 나와서 2가 아예 안나옴.

earlyStopping= EarlyStopping(monitor='val_loss',patience=10,mode='min',restore_best_weights=True,verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=100,validation_split=0.2,callbacks=[earlyStopping,],verbose=1) #batch default :32



#4.평가,예측
results = model.evaluate(x_test,y_test)
print('loss : ', results[0])

y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict,axis=1) 

y_test = tf.argmax(y_test,axis=1) 
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)


#만들기! acc 0.98이상

