# 과제
# activation : sigmoid, relu, linear 넣어라.
# metrics 추가
# earlyStopping 넣구 성능비교
# 느낀줄 2줄이상!!

#activation(활성화 함수): 모든레이어에 강림하신다. 한정시키는 역할 
#계단함수(0.1만 있는것)>>> 시그모이드함수(0과1사이의 실수를 한정하여 출력하는 것,음수는 0으로 취급)>>>
import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from sklearn.datasets import load_boston
#1.데이터
datasets = load_boston()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape) #(506,13)
#2.모델
model = Sequential()
model.add(Dense(50,activation = 'relu', input_dim=13))
model.add(Dense(50,activation ='relu'))
model.add(Dense(50,activation ='relu'))
model.add(Dense(50,activation ='relu'))
model.add(Dense(50,activation ='relu'))
model.add(Dense(1,activation ='linear'))

#3.컴파일,훈련 
model.compile(loss='mse',optimizer='adam',metrics='accuracy')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor= 'val_loss',patience=20,mode='min',restore_best_weights=True,verbose=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, shuffle=True,random_state=31)

model.fit(x_train,y_train,epochs=1000,batch_size=10,verbose=1,validation_split= 0.2,callbacks=earlyStopping)

#4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

#R2결정계수(성능평가지표)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 

# loss:  [22.193241119384766, 0.0]
# r2스코어 :  0.7301849606079567

#회귀모델에서는 acc 나오지않았다. relu,early stopping을 썼을 때 성능이 상승했다. patience 횟수는 10,20,50 다 비슷했다.
