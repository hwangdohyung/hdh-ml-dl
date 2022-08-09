# [실습]
# 아까 4가지로 모델 만들기 
# 784개 DNN으로 만든거(최상의 성능인거 // 0.996이상)과 비교!!

# time 체크 / fit에서 하고 

#1. 나의 최고의 DNN 
# time = ???
# acc = ???

#2. 나의 최고의 CNN 
# time = ???
# acc = ???

#3. PCA 0.95 
# time = ???
# acc = ???

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np 
from sklearn.decomposition import PCA
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
import time
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.callbacks import EarlyStopping

#1.데이터
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape,y_train.shape,y_test.shape)

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

x = x.reshape(70000,784)

pca = PCA(n_components=713)   
x= pca.fit_transform(x) 

pca_EVR = pca.explained_variance_ratio_ 

cumsum = np.cumsum(pca_EVR)

x_train_cnn,x_test_cnn,y_train_cnn,y_test_cnn = train_test_split(x,y , train_size=0.8, random_state=123, shuffle=True)


#(60000, 28, 28) (10000, 28, 28) (60000,) (10000,)

# ################## CNN ####################
# x_train_cnn = x_train_cnn.reshape(60000,28,28,1)
# x_test_cnn = x_test_cnn.reshape(10000,28,28,1)

# from tensorflow.keras.utils import to_categorical
# y_train_cnn = to_categorical(y_train_cnn)
# y_test_cnn = to_categorical(y_test_cnn)

# #2.모델구성
# model_cnn = Sequential()
# model_cnn.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', input_shape=(28,28,1))) 
# model_cnn.add(MaxPooling2D())
# model_cnn.add(Conv2D(7, (2,2), padding='valid', activation='relu'))
# model_cnn.add(Conv2D(7, (2,2), padding='valid', activation='relu'))
# model_cnn.add(Conv2D(7, (2,2), padding='valid', activation='relu'))
# model_cnn.add(Flatten())
# model_cnn.add(Dense(32, activation='relu'))
# model_cnn.add(Dense(32, activation='relu'))
# model_cnn.add(Dense(10, activation='softmax')) 
# model_cnn.summary()

# start_cnn = time.time()
# #3.컴파일 훈련
# model_cnn.compile(loss='categorical_crossentropy', optimizer='adam')
# model_cnn.fit(x_train_cnn, y_train_cnn, epochs=50, batch_size=20)
# end_cnn = time.time()

# ############### DNN ###############


x_train_dnn,x_test_dnn,y_train_dnn,y_test_cnn = train_test_split(x,y , train_size=0.8, random_state=123, shuffle=True)
print(x_train_dnn.shape,y_train_dnn.shape)




#2. 모델구성
model_dnn =Sequential()
model_dnn.add(Dense(64,activation='relu', input_shape = (713,)))   #위와 동일
model_dnn.add(Dropout(0.2))
model_dnn.add(Dense(32,activation = 'relu'))
model_dnn.add(Dense(32, activation='relu'))
model_dnn.add(Dropout(0.2))
model_dnn.add(Dense(16, activation='relu'))
model_dnn.add(Dense(10, activation='softmax')) 

# 3.컴파일,훈련
model_dnn.compile(loss='categorical_crossentropy', optimizer='adam')

earlyStopping = EarlyStopping(monitor= 'val_loss',patience=10,mode='min',restore_best_weights=True,verbose=1)

start_dnn = time.time()
hist = model_dnn.fit(x_train_dnn,y_train_cnn,epochs=50,batch_size=32,verbose=1,validation_split= 0.2,callbacks=[earlyStopping])
end_dnn = time.time()


################## PCA ###################
# x = np.append(x_train, x_test, axis=0)
# y = np.append(y_train, y_test, axis=0)

# x = x.reshape(70000,784)

# pca = PCA(n_components=713)   
# x= pca.fit_transform(x) 

# pca_EVR = pca.explained_variance_ratio_ 

# cumsum = np.cumsum(pca_EVR)

# x_train,x_test,y_train,y_test = train_test_split(x,y , train_size=0.8, random_state=123, shuffle=True)

# #2.모델
# from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier()

# #3.훈련 
# start_pca = time.time()
# model.fit(x_train, y_train) #, eval_metric= 'error')
# end_pca = time.time()

# ###############################################################
# ## pca 예측 ## 
# results = model.score(x_test, y_test)
# print('=================== pca ====================')
# print('결과 : ', results)
# print('걸린시간cnn : ', round(end_pca - start_pca, 2))

## dnn 예측 ##
loss_dnn = model_dnn.evaluate(x_test_dnn, y_test_cnn)

y_predict_dnn = model_dnn.predict(x_test_dnn) 

y_predict_dnn = np.argmax(y_predict_dnn, axis= 1)
y_test_dnn = np.argmax(y_test_cnn, axis= 1)
acc_dnn = accuracy_score(y_test_dnn, y_predict_dnn) 
print('=================== dnn ====================')
print('acc스코어: ', acc_dnn)
print('걸린시간dnn : ', round(end_dnn - start_dnn,2))

# ## cnn 예측 ##
# loss_cnn = model_cnn.evaluate(x_test_cnn, y_test_cnn)
# y_predict_cnn = model_cnn.predict(x_test_cnn)

# y_predict_cnn = np.argmax(y_predict_cnn, axis= 1)
# y_test_cnn = np.argmax(y_test_cnn, axis= 1)

# from sklearn.metrics import accuracy_score
# acc_cnn = accuracy_score(y_test_cnn, y_predict_cnn)
# print('=================== cnn ====================')
# print('acc스코어: ', acc_cnn)
# print('걸린시간cnn : ', round(end_cnn - start_cnn, 2))


# =================== pca1.0====================
# 결과 :  0.9051428571428571
# 걸린시간cnn :  163.15
# =================== pca0.999====================
# 결과 :  0.9180714285714285
# 걸린시간cnn :  131.92
# =================== pca0.99 ====================
# 결과 :  0.9387142857142857
# 걸린시간cnn :  102.84
# =================== pca0.95 ====================
# 결과 :  0.9457142857142857
# 걸린시간cnn :  66.93
# =================== dnn ====================
# acc스코어:  0.962
# 걸린시간dnn :  240.66
# =================== cnn ====================
# acc스코어:  0.9818
# 걸린시간cnn :  587.2