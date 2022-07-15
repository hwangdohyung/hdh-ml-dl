#1.데이터 
from re import escape
import numpy as np
from sklearn.metrics import r2_score 
x1_datasets = np.array([range(100),range(301,401)])  #삼성전자 종가 , 하이닉스 종가 
x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]) # 원유, 돈육 , 밀 
x3_datasets = np.array([range(100,200),range(1301,1401)]) # 우리반 아이큐, 우리반 키 

x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)
x3 = np.transpose(x3_datasets)

print(x1.shape,x2.shape,x3.shape) #(100,2) (100,3) (100,2)

y1 = np.array(range(2001,2101))  # 금리
y2 = np.array(range(201,301))  # 환율


from sklearn.model_selection import train_test_split

x1_train,x1_test,x2_train,x2_test,x3_train,x3_test,y1_train,y1_test,y2_train,y2_test = train_test_split(x1,x2,x3,y1,y2, train_size=0.7,random_state=77)

print(x1_train.shape,x1_test.shape) 
print(x2_train.shape,x2_test.shape)
print(x3_train.shape,x3_test.shape) 
print(y1_train.shape,y1_test.shape) 
print(y2_train.shape,y2_test.shape) 


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

#2-3 모델3
input3 = Input(shape=(2,))
dense21 = Dense(100, activation='relu', name='dh21')(input3) 
dense22 = Dense(100, activation='relu', name='dh22')(dense21) 
dense23 = Dense(100, activation='relu', name='dh23')(dense22) 
dense24 = Dense(100, activation='relu', name='dh24')(dense23) 
output3 = Dense(10, name='out_dh3')(dense24) 

# concatenate
from tensorflow.python.keras.layers import concatenate,Concatenate  # concatenate - 사슬처럼 엮다. 단순하게 이어버리는것. 다른모델도 많다(가중치를 합치거나 연산하는 등)
# merge1 = concatenate([output1,output2,output3],name='mg1') # 두개이상은 리스트! , output1(10)뒤에 output2 (10) 이 이어져 있는 denselayer
merge1 = Concatenate()([output1,output2,output3])
merge2 = Dense(50, activation='relu',name='mg2')(merge1)
merge3 = Dense(50,name='mg3')(merge2)
last_output = Dense(1,name='last')(merge3)

#output 모델 1
output41 = Dense(50, activation='relu')(last_output)
output42 = Dense(50,)(output41)
output43 = Dense(50,)(output42)
last_output2 = Dense(1,name='last2')(output43)

#output 모델 2
output51 = Dense(50, activation='relu')(last_output)
output52 = Dense(50,)(output51)
output53 = Dense(50,)(output52)
last_output3 = Dense(1,name='last3')(output53)

model = Model(inputs=[input1,input2,input3],outputs=[last_output2,last_output3])

#3.컴파일 훈련
model.compile(loss='mae', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor= 'val_loss',patience=30,mode='min',verbose=1,restore_best_weights=True)
model.fit([x1_train,x2_train,x3_train], [y1_train,y2_train], epochs=100, batch_size=1,validation_split=0.1,callbacks= es)

#4.평가,예측
loss = model.evaluate([x1_test,x2_test,x3_test],[y1_test,y2_test])
y1_predict,y2_predict = model.predict((x1_test,x2_test,x3_test))

print('loss: ', loss)   
#loss:  [52.99702453613281, 30.54537010192871, 22.451656341552734] 3개 나오는이유 loss는 상대적지표이기 때문 이전것과 비교하기 때문  뒤에 2개 합한것이 앞에꺼
from sklearn.metrics import r2_score

r2_1= r2_score(y1_test,y1_predict)
r2_2= r2_score(y2_test,y2_predict)

print('r2_1: ', r2_1)
print('r2_2: ', r2_2)

# loss:  [44.92525863647461, 7.857348442077637, 37.067909240722656]
# r2_1:  0.867066048237226
# r2_2:  -1.6795586084104834

