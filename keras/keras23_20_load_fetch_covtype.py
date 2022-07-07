import numpy as np 
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

import pandas as pd

#1.데이터 
datasets = fetch_covtype()
x= datasets.data
y= datasets.target

print(x.shape, y.shape) #(581012, 54) (581012, )
print(np.unique(y,return_counts=True)) #(array[1 2 3 4 5 6 7],array[211840, 283301,  35754,   2747,   9493,  17367,  20510] )

#텐서플로우
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y) 

# 판다스 겟더미
# y= pd.get_dummies(y) #argmax 다르게 하니 돌아감. 이유는 모름

#사이킷런
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categories='auto',sparse= False)#False로 할 경우 넘파이 배열로 반환된다.
y = y.reshape(-1,1)
one.fit(y)
y = one.transform(y)

print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=True,
                                                     random_state=58)

# minmax , standard
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 

#2.모델
model = Sequential()
model.add(Dense(40, input_dim=54))
model.add(Dense(30,activation ='relu'))
model.add(Dense(30,activation ='relu'))
model.add(Dense(30,activation ='relu'))
model.add(Dense(30,activation ='relu'))
model.add(Dense(7,activation ='softmax'))

model.load_weights('./_save/keras23_19_save_fetch_covtype.h5')          

#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer ='adam', metrics='accuracy') #다중분류는 무조건 loss에 categorical_crossentropy
 #분류모델에서 셔플 중요! ,false로 하면 순차적으로 나와서 2가 아예 안나옴.
 
 #4.평가,예측
# loss,acc = model.evaluate(x_test,y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)
#################### 위와 동일###############
results = model.evaluate(x_test,y_test)
print('loss : ', results[0])
# print('accuracy : ', results[1])
############################################

# print(y_test)
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict,axis=1) 

# print(y_test)
# print(y_test.shape)

y_test = tf.argmax(y_test,axis=1) 
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)