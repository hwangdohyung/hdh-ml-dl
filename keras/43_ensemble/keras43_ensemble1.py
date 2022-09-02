#1.데이터 
from re import escape
import numpy as np
from sklearn.metrics import r2_score 
x1_datasets = np.array([range(100),range(301,401)])  #삼성전자 종가 , 하이닉스 종가 
x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]) # 원유, 돈육 , 밀 
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)

print(x1.shape,x2.shape)  #(100,2) (100,3)
y = np.array(range(2001,2101))  # 금리

from sklearn.model_selection import train_test_split

x1_train,x1_test,x2_train,x2_test,y_train,y_test = train_test_split(x1,x2,y, train_size=0.7,random_state=77)

print(x1_train.shape,x1_test.shape,x2_train.shape,x2_test.shape,y_train.shape,y_test.shape) #(70, 2) (30, 2) (70, 3) (30, 3) (70,) (30,)

#2. 모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense,Input

#2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu', name='dh1')(input1) 
dense2 = Dense(100, activation='relu', name='dh2')(dense1) 
dense3 = Dense(100, activation='relu', name='dh3')(dense2) 
output1 = Dense(10, name='out_dh1')(dense3) 

#2-2 모델2
input2 = Input(shape=(3,))
dense11 = Dense(100, activation='relu', name='dh11')(input2) 
dense12 = Dense(100, activation='relu', name='dh12')(dense11) 
dense13 = Dense(100, activation='relu', name='dh13')(dense12) 
dense14 = Dense(100, activation='relu', name='dh14')(dense13) 
output2 = Dense(10, name='out_dh2')(dense14) 

from tensorflow.python.keras.layers import concatenate,Concatenate   # concatenate - 사슬처럼 엮다. 단순하게 이어버리는것. 다른모델도 많다(가중치를 합치거나 연산하는 등)
# concatenate 함수형(소문자c) *쓰는방식이 다르다!
merge1 = concatenate([output1,output2],name='mg1') # 두개이상은 리스트! , output1(10)뒤에 output2 (10) 이 이어져 있는 denselayer
merge2 = Dense(2, activation='relu',name='mg2')(merge1)
merge3 = Dense(3,name='mg3')(merge2)
last_output = Dense(1,name='last')(merge3)

model = Model(inputs=[input1,input2],outputs=last_output)

model.summary()
# 섞어서 연산한다. concat은 연산량 0

# Model: "model"
# __________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected
# ==========================================================================
# input_2 (InputLayer)            [(None, 3)]          0
# __________________________________________________________________________
# input_1 (InputLayer)            [(None, 2)]          0
# __________________________________________________________________________
# dh11 (Dense)                    (None, 11)           44          input_2[0
# __________________________________________________________________________
# dh1 (Dense)                     (None, 1)            3           input_1[0
# __________________________________________________________________________
# dh12 (Dense)                    (None, 12)           144         dh11[0][0
# __________________________________________________________________________
# dh2 (Dense)                     (None, 2)            4           dh1[0][0]
# __________________________________________________________________________
# dh13 (Dense)                    (None, 13)           169         dh12[0][0
# __________________________________________________________________________
# dh3 (Dense)                     (None, 3)            9           dh2[0][0]
# __________________________________________________________________________
# dh14 (Dense)                    (None, 14)           196         dh13[0][0
# __________________________________________________________________________
# out_dh1 (Dense)                 (None, 10)           40          dh3[0][0]
# __________________________________________________________________________
# out_dh2 (Dense)                 (None, 10)           150         dh14[0][0
# __________________________________________________________________________
# mg1 (Concatenate)               (None, 20)           0           out_dh1[0
#                                                                  out_dh2[0
# __________________________________________________________________________
# mg2 (Dense)                     (None, 2)            42          mg1[0][0]
# __________________________________________________________________________
# mg3 (Dense)                     (None, 3)            9           mg2[0][0]
# __________________________________________________________________________
# last (Dense)                    (None, 1)            4           mg3[0][0]
# ==========================================================================
# Total params: 814
# Trainable params: 814
# Non-trainable params: 0

#3.컴파일 훈련
model.compile(loss='mae', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor= 'val_loss',patience=10,mode='min',verbose=1,restore_best_weights=True)

model.fit([x1_train,x2_train], y_train, epochs=200, batch_size=1,validation_split=0.1,callbacks= es)


#4.평가,예측
loss = model.evaluate([x1_test,x2_test], y_test)
y_predict = model.predict([x1_test,x2_test])
print('loss: ', loss)


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 

# loss:  10.349862098693848
# r2스코어 :  0.8417548799292373




