#다중분류 point) --- loss categorical!, softmax ,마지막 노드갯수!,one hot encoding
import numpy as np 
from sklearn.datasets import load_iris
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Input,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import tensorflow as tf

#1.데이터
datasets = load_iris()

x= datasets['data']
y= datasets['target']
# print(x)
print(y)
# print(x.shape,y.shape) #(150,4) (150, )


#분류모델은 모델 전에 one hot encoding 필수(전처리과정)
#################### one hot encoding ####################### 2가지 방법
print('y의 라벨값 : ', np.unique(y,return_counts=True)) #무슨값으로 이루어져 있는지 확인하는것(0,1,2) #return:각각 몇개인지 확인

#텐서플로우
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)
###################

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=True,
                                                     random_state=58)

# minmax , standard
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 

#2.모델구성
input1= Input(shape= (4,))
dense1= Dense(40,activation='relu')(input1)
drop1 = Dropout(0.3)(dense1)
dense2= Dense(40,activation='relu')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3= Dense(40,activation='relu')(drop2)
dense4= Dense(40,activation='relu')(dense3)
dense5= Dense(40,activation='relu')(dense4)
output1 = Dense(3,activation='softmax')(dense5)
model=Model(inputs=input1,outputs=output1)

import datetime 
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')



#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer ='adam', metrics='accuracy') #다중분류는 무조건 loss에 categorical_crossentropy
 #분류모델에서 셔플 중요! ,false로 하면 순차적으로 나와서 2가 아예 안나옴.

earlyStopping= EarlyStopping(monitor='val_loss',patience=30,mode='min',restore_best_weights=True,verbose=1)

# filepath = './_ModelCheckpoint/k24/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                    #   filepath="".join([filepath,'iris',date,'_',filename]))

model.fit(x_train, y_train, epochs=1000, batch_size=32,validation_split=0.2,callbacks=[earlyStopping] ,verbose=1)

#4.평가,예측

results = model.evaluate(x_test,y_test)
print('loss : ', results[0])


y_predict = model.predict(x_test) # x값 4번째까지
y_predict = y_predict.argmax(axis=1) # 최대값의 위치 구해줌. argmin은 최솟값 (n, 3)에서(n, 1)로 변경됨.


y_test = y_test.argmax(axis=1) # y_test 값도 최대값 추출해줘야함 (n, 3) 에서 (n, 1)로 변경 
acc = accuracy_score(y_test,y_predict)# acc 정수값을 원한다. 
print('acc : ',acc)

# loss :  0.06750369817018509
# acc :  0.9777777777777777