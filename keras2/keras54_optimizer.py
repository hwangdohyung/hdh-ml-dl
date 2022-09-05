import tensorflow as tf 
import numpy as np 

# 1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,7,11,9,7])

# 2.모델 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim= 1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

# 3.컴파일,훈련 
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax
from tensorflow.python.keras.optimizer_v2 import rmsprop, nadam 

learning_rate = 0.001
op_list = [adam.Adam,adadelta.Adadelta,adagrad.Adagrad,adamax.Adamax,rmsprop.RMSprop,nadam.Nadam]
op_name = ['Adam','Adadelta','Adagrad','Adamax','RMSprop','Nadam']
result = []
for i,n in zip(op_list,op_name):
    optimizer = i(lr=learning_rate)
    model.compile(loss= 'mse', optimizer=optimizer)
    model.fit(x,y, epochs= 50, batch_size=1,verbose=0)

    # 4.평가,예측
    loss = model.evaluate(x,y)
    y_predict = model.predict([11])

    re = n,':loss : ', loss, 'lr : ', learning_rate, '결과 : ', y_predict
    result.append(re)
print('=============================================')    
print(result)

# [('Adam', ' : ', 'loss : ', 2.3268134593963623, 'lr : ', 0.001, '결과 : ', array([[11.241889]], dtype=float32)),
# ('Adadelta', ' : ', 'loss : ', 2.2175068855285645, 'lr : ', 0.001, '결과 : ', array([[10.951665]], dtype=float32)), 
# ('Adagrad', ' : ', 'loss : ', 2.1617014408111572, 'lr : ', 0.001, '결과 : ', array([[10.421021]], dtype=float32)), 
# ('Adamax', ' : ', 'loss : ', 2.1629791259765625, 'lr : ', 0.001, '결과 : ', array([[10.375107]], dtype=float32)), 
# ('RMSprop', ' : ', 'loss : ', 2.5738155841827393, 'lr : ', 0.001, '결과 : ', array([[11.621265]], dtype=float32)), 
# ('Nadam', ' : ', 'loss : ', 2.644552707672119, 'lr : ', 0.001, '결과 
# : ', array([[9.265449]], dtype=float32))]

