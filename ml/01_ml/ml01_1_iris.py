#다중분류 point) --- loss categorical!, softmax ,마지막 노드갯수!,one hot encoding
import numpy as np 
from sklearn.datasets import load_iris
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC


import tensorflow as tf
tf.random.set_seed(66)

#1.데이터
datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)

x= datasets['data']
y= datasets['target']


# #################### one hot encoding ####################### 
# print('y의 라벨값 : ', np.unique(y,return_counts=True)) 

# #텐서플로우
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=True,
                                                     random_state=58) 

# #2.모델
# model = Sequential()
# model.add(Dense(10, input_dim=4))
# model.add(Dense(10,activation ='relu'))
# model.add(Dense(10,activation ='relu'))
# model.add(Dense(10,activation ='relu'))
# model.add(Dense(10,activation ='relu'))
# model.add(Dense(3,activation ='softmax'))
model = LinearSVC()


#3.컴파일,훈련
# model.compile(loss= 'categorical_crossentropy', optimizer ='adam', metrics='accuracy') 
#
# earlyStopping= EarlyStopping(monitor='val_loss',patience=30,mode='min',restore_best_weights=True,verbose=1)

# model.fit(x_train, y_train, epochs=1000, batch_size=1,validation_split=0.2,callbacks=earlyStopping, verbose=1)

model.fit(x_train,y_train)

#4.평가,예측
# loss,acc = model.evaluate(x_test,y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)
#################### 위와 동일###############
# results = model.evaluate(x_test,y_test)
# print('loss : ', results[0])
# print('accuracy : ', results[1])
############################################
result = model.score(x_test, y_test)

print('결과 acc : ', result)
y_predict = model.predict(x_test) 
# y_predict = y_predict.argmax(axis=1) 

# y_test = y_test.argmax(axis=1) 
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)

# print(y_predict)
# print(y_test)


