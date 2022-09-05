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
for i in op_list:
    optimizer = i(lr=learning_rate)
    model.compile(loss= 'mse', optimizer=optimizer)
    model.fit(x,y, epochs= 50, batch_size=1,verbose=0)

    # 4.평가,예측
    loss = model.evaluate(x,y)
    y_predict = model.predict([11])

    re = 'loss : ', loss, 'lr : ', learning_rate, '결과 : ', y_predict
    result.append(re)
print('=============================================')    
print(result)

# [('loss : ', 2.392810344696045, 'lr : ', 0.001, '결과 : ', array([[11.34106]], dtype=float32)), 
# ('loss : ', 2.234635353088379, 'lr : ', 0.001, '결과 : ', array([[10.993814]], dtype=float32)), 
# ('loss : ', 2.1593987941741943, 'lr : ', 0.001, '결과 : ', array([[10.514917]], dtype=float32)), 
# ('loss : ', 2.1637518405914307, 'lr : ', 0.001, '결과 : ', array([[10.349235]], dtype=float32)), 
# ('loss : ', 2.172139883041382, 'lr : ', 0.001, '결과 : ', array([[10.230819]], dtype=float32)), 
# ('loss : ', 2.2593631744384766, 'lr : ', 0.001, '결과 : ', array([[9.909864]], dtype=float32))]

