# 과제
# activation : sigmoid, relu, linear 넣어라.
# metrics 추가
# earlyStopping 넣구 성능비교
# 느낀줄 2줄이상!!
from lightgbm import train
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
model.add(Dense(1,activation =''))

#3.컴파일,훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, shuffle=True,random_state=31)

model.fit(x_train,y_train,epochs=1000,batch_size=10,verbose=1,validation= 0.2,)