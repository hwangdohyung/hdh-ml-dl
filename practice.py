import numpy as np
from numpy.testing._private.utils import import_nose

#1. 데이터
x1 = np.array([range(100), range(301, 401)])       
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)]) 
x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.array(range(1001, 1101))  
y2 = np.array(range(101, 201)) 
y3 = np.array(range(401, 501)) 

print(x1.shape, x2.shape, y1.shape, y2.shape)  # (100, 2) (100, 3) (100,) (100,)



from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, x2, y1, y2, train_size=0.7, random_state=66)
y3_train, y3_test = train_test_split(y3, train_size=0.7, random_state=66)

print(x1_train.shape, x1_test.shape)  # (70, 2) (30, 2)
print(x2_train.shape, x2_test.shape)  # (70, 3) (30, 3)
print(y1_train.shape, y1_test.shape)    # (70,) (30,)
print(y2_train.shape, y2_test.shape)    # (70,) (30,)

#2. 모델구성
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

#2-1 모델 1
input1 = Input(shape=(2,))
dense1 = Dense(5, name='dense1', activation='relu')(input1)
dense2 = Dense(7, name='dense2', activation='relu')(dense1)
dense3 = Dense(7, name='dense3', activation='relu')(dense2)
output1 = Dense(7, name='output1', activation='relu')(dense3)

#2-2 모델 2
input2 = Input(shape=(3,))
dense11 = Dense(10, name='dense11', activation='relu')(input2)
dense12 = Dense(10, name='dense12', activation='relu')(dense11)
dense13 = Dense(10, name='dense13', activation='relu')(dense12)
dense14 = Dense(10, name='dense14', activation='relu')(dense13)
output2 = Dense(10, name='output2', activation='relu')(dense14)

#앙상블
from tensorflow.python.keras.layers import Concatenate, concatenate
merge1 = concatenate([output1, output2])

#2-3 out모델 1
output21 = Dense(7, activation='relu')(merge1)
output22 = Dense(11)(output21)
output23 = Dense(11)(output22)
last_output1 = Dense(1)(output23)

#2-4 out모델 2
output31 = Dense(7, activation='relu')(merge1)
output32 = Dense(21)(output31)
output33 = Dense(21)(output32)
output34 = Dense(21)(output33)
last_output2 = Dense(1)(output34)

#2-5 out모델 3
output41 = Dense(7, activation='relu')(merge1)
output42 = Dense(21)(output41)
output43 = Dense(21)(output42)
output44 = Dense(21)(output43)
last_output3 = Dense(1)(output44)

model = Model(inputs=[input1, input2], outputs=[last_output1, last_output2, last_output3])

model.summary()

#3. 컴파일
model.compile(loss='mae', optimizer='adam', metrics='mse')
hist = model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train], epochs=10, batch_size=1)

#4. 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test, y3_test])
print("loss : ", loss)

y_predict = model.predict([x1_test,x2_test])

from sklearn.metrics import r2_score
# r2= r2_score([y1_test,y2_test,y3_test],y_predict)
# print('r2: ', r2)
