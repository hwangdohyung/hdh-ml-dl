from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score

#1.데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR) #(569, 30)
# print(datasets.feature_names)

x = datasets['data']
y = datasets['target']
print(x.shape, y.shape)

#2.모델구성
model = Sequential()
model.add(Dense(10,activation='linear', input_dim = 30)) #linear: 디폴트값, 선형회귀 
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu')) # 성능이 좋은 relu 히든에서만 쓸수 있다.
model.add(Dense(10,activation='linear'))
model.add(Dense(10,activation='linear'))
model.add(Dense(1,activation='sigmoid'))# 회귀모델은 ouuput에 linear,2진분류는 무조건 마지막 output에 activation은 sigmoid(0과1사이의 값으로나온다. 사이값은 반올림하면 된다.)

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping= EarlyStopping(monitor= 'val_loss',patience=80,mode='min',restore_best_weights=True,verbose=1) 

#3.컴파일,훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy','mse'],) #모델과 예측값을 비교하는곳 2진분류는 무조건 binary쓴다. 다중분류는 softmax 
                                                                  #리스트 형태 평가지표 2개이상 계속 넣을수 있다. mse는 2진분류에서 신뢰할수없다.              
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)
hist=model.fit(x_train,y_train,epochs=100, batch_size=32,verbose=1,validation_split=0.2, callbacks= [earlyStopping])#callback 리스트형태 더 호출할수있다.

#4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss: ', loss)

y_predict = model.predict(x_test).round() # 반올림을 해줘야 acc가 나온다 
# y_predict = y_predict.round()
acc = accuracy_score(y_test, y_predict) 
print('acc스코어: ', acc)
print(y_predict)

################ 분류모델에서는 r2 말고 acc쓴다. ###################
