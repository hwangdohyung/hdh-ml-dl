from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input,Dropout
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint

#1.데이터
datasets = load_breast_cancer()

x = datasets['data']
y = datasets['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)

# # minmax , standard
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
input1 = Input(shape=(30,))
dense1 = Dense(50,activation= 'relu')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(50,activation= 'relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(50,activation= 'relu')(drop2)
dense4 = Dense(40,activation= 'relu')(dense3)
dense5 = Dense(30,activation= 'relu')(dense4)
output1 = Dense(1,activation='sigmoid')(dense5)
model= Model(inputs=input1,outputs=output1)

import datetime 
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

earlyStopping= EarlyStopping(monitor= 'val_loss',patience=30,mode='min',restore_best_weights=True,verbose=1) 


#3.컴파일,훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy','mse'],) 

# filepath ='./_ModelCheckpoint/k24/'
# filename ='{epoch:04d}-{val_loss:.4f}.hdf5'
                                                                             
# mcp = ModelCheckpoint(monitor = 'val_loss',mode= 'auto',verbose=1,save_best_only=True,
#                       filepath = "".join([filepath,'cancer',date,'_',filename]))

hist=model.fit(x_train,y_train,epochs=1000, batch_size=32,verbose=1,validation_split=0.2, callbacks= [earlyStopping])


#4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss: ', loss)

y_predict = model.predict(x_test) 
y_predict = y_predict.round()
acc = accuracy_score(y_test, y_predict) 
print('acc스코어: ', acc)

# loss:  [0.03570733591914177, 0.9941520690917969, 0.006787752732634544]
# acc스코어:  0.9941520467836257
