import tensorflow as tf 
import numpy as np 
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2.모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(28,28,1))) 
model.add(MaxPooling2D())
model.add(Conv2D(7, (2,2), padding='valid', activation='relu'))
model.add(Conv2D(7, (2,2), padding='valid', activation='relu'))
model.add(Conv2D(7, (2,2), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) 
model.summary()

from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax
from tensorflow.python.keras.optimizer_v2 import rmsprop, nadam 

#3.컴파일 훈련

learning_rate = 0.001

op_list = [adam.Adam,adadelta.Adadelta,adagrad.Adagrad,adamax.Adamax,rmsprop.RMSprop,nadam.Nadam]
op_name = ['Adam','Adadelta','Adagrad','Adamax','RMSprop','Nadam']
result = []

for i,n in zip(op_list,op_name):
    

    optimizer = i(lr= learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.fit(x_train, y_train, epochs=1, batch_size=20,verbose=0)

    #4.평가 훈련
    loss = model.evaluate(x_test, y_test)
    y_predict = model.predict(x_test)
    y_predict = np.argmax(y_predict, axis= 1)
    y_test = np.argmax(y_test, axis= 1)
    

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_predict)
    re = n,':loss : ', loss, 'lr : ', learning_rate, '결과 : ', y_predict, 'acc : ',acc
    result.append(re)
    y_test = to_categorical(y_test)
print('=============================================')    
print(result)
    

# [('Adam', ':loss : ', 0.09247702360153198, 'lr : ', 0.001, '결과 : ', array([7, 2, 1, ..., 4, 5, 6], dtype=int64), 'acc : ', 0.9712), 
# ('Adadelta', ':loss : ', 0.0838637426495552, 'lr : ', 0.001, '결과 : ', array([7, 2, 1, ..., 4, 5, 6], dtype=int64), 'acc : ', 0.9746), 
# ('Adagrad', ':loss : ', 0.0580800361931324, 'lr : ', 0.001, '결과 : ', array([7, 2, 1, ..., 4, 5, 6], dtype=int64), 'acc : ', 0.9813), 
# ('Adamax', ':loss : ', 0.05223749205470085, 'lr : ', 0.001, '결과 : ', array([7, 2, 1, ..., 4, 5, 6], dtype=int64), 'acc : ', 0.9851), 
# ('RMSprop', ':loss : ', 0.0676237940788269, 'lr : ', 0.001, '결과 : ', array([7, 2, 1, ..., 4, 5, 6], dtype=int64), 'acc : ', 0.979), 
# ('Nadam', ':loss : ', 0.07385706901550293, 'lr : ', 0.001, '결과 : ', array([7, 2, 1, ..., 4, 5, 6], dtype=int64), 'acc : ', 0.9772)]


