import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,20,30,40,50,60,70,80,90,100])
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(1))
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7, shuffle=True,random_state=67)
model.compile(loss ='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size= 1, verbose = 2)
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
print('y예측값 : ', y_predict)
r2 = r2_score(y_predict,y_test)
print('r2 : ', r2)