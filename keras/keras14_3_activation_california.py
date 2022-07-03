import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


#1.데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

#2.모델구성
model = Sequential()
model.add(Dense(50, input_dim=8)) 
model.add(Dense(50,activation ='relu'))
model.add(Dense(50,activation ='relu'))
model.add(Dense(50,activation ='relu'))
model.add(Dense(50,activation ='relu'))
model.add(Dense(50,activation ='relu'))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping

earlyStopping = EarlyStopping(monitor= 'val_loss',patience=30,mode= 'min', restore_best_weights=True,verbose=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100)

hist = model.fit(x_train, y_train, epochs=2000, batch_size=50,
         validation_split= 0.2,callbacks= earlyStopping)

#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
y_predict = y_predict.round()
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 