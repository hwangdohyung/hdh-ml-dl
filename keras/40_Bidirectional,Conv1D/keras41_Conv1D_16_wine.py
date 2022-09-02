import numpy as np 
from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Input,Dropout,LSTM,Conv1D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

#1.데이터 
datasets = load_wine()
x= datasets.data
y= datasets.target

print(x.shape, y.shape) #(178, 13) (178, )
print(np.unique(y,return_counts=True)) #y의 라벨 ,[0 1 2]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=True,
                                                     random_state=58)

# minmax , standard

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape,x_test.shape)
x_train = x_train.reshape(124,13,1)
x_test = x_test.reshape(54,13,1)



#2.모델구성
model = Sequential()
model.add(Conv1D(200,2,activation='relu', input_shape=(13,1))) 
model.add(Flatten())
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(3,activation='softmax'))


import datetime 
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer ='adam', metrics='accuracy') 
 
earlyStopping= EarlyStopping(monitor='val_loss',patience=10,mode='min',restore_best_weights=True,verbose=1)

# filepath = './_ModelCheckpoint/k24/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss',node='auto',save_best_only=True,verbose=1,
#                       filepath="".join([filepath,'wine',date,'_',filename]))

model.fit(x_train, y_train, epochs=100, batch_size=32,validation_split=0.2,callbacks=[earlyStopping], verbose=1)



#4.평가,예측

results = model.evaluate(x_test,y_test)
print('loss : ', results[0])

y_predict = model.predict(x_test) 
y_predict = y_predict.argmax(axis=1) 


y_test = y_test.argmax(axis=1) 
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)

#DNN
# loss :  0.16414682567119598
# acc :  0.9444444444444444

#LSTM
# loss :  0.22602766752243042
# acc :  0.9074074074074074

#Conv1D
# loss :  0.09341952204704285
# acc :  0.9814814814814815