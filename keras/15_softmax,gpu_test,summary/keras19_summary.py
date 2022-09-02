from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np 

#1.데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2.모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation= 'relu'))
model.add(Dense(4, activation= 'sigmoid'))
model.add(Dense(3, activation= 'relu'))
model.add(Dense(1))

model.summary() 
#_________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 5)                 10                    y= wx +b
# _________________________________________________________________           
# dense_1 (Dense)              (None, 3)                 18
# _________________________________________________________________
# dense_2 (Dense)              (None, 4)                 16
# _________________________________________________________________
# dense_3 (Dense)              (None, 3)                 15
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 4
# =================================================================
# Total params: 63
# Trainable params: 63
# Non-trainable params: 0
# _________________________________________________________________
# y=wx+b 라서 param에 bias노드갯수 추가됨.