import numpy as np 
from sklearn.datasets import load_digits
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Input,Dropout,LSTM,Conv1D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

#1.데이터 
datasets = load_digits()
x= datasets.data
y= datasets.target

print(x.shape, y.shape) #(1797, 64) (1797, ) * 64개의 컬럼이 아니라 8x8 이미지 (64칸) 그게 1797개가 있다.
print(np.unique(y,return_counts=True)) #y의 라벨 ,[0 1 2 3 4 5 6 7 8 9]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=True,
                                                     random_state=58)

# minmax , standard
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 

print(x_train.shape,x_test.shape)
x_train = x_train.reshape(1257,64,1)
x_test = x_test.reshape(540,64,1)



#2.모델구성
model = Sequential()
model.add(Conv1D(200,2,activation='relu', input_shape=(64,1))) 
model.add(Flatten())
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(10,activation='softmax'))


#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer ='adam', metrics='accuracy') #다중분류는 무조건 loss에 categorical_crossentropy
 #분류모델에서 셔플 중요! ,false로 하면 순차적으로 나와서 2가 아예 안나옴.

earlyStopping= EarlyStopping(monitor='val_loss',patience=10,mode='min',restore_best_weights=True,verbose=1)

# filepath = './_ModelCheckpoint/k24/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
#                       filepath="".join([filepath,'diabets',date,'_',filename]))

model.fit(x_train, y_train, epochs=100, batch_size=64,validation_split=0.2,callbacks=[earlyStopping], verbose=1)


#4.평가,예측
# loss,acc = model.evaluate(x_test,y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)
#################### 위와 동일###############
results = model.evaluate(x_test,y_test)
print('loss : ', results[0])
# print('accuracy : ', results[1])
############################################

# print(y_test)
y_predict = model.predict(x_test) 
y_predict = y_predict.argmax(axis=1) 

y_test = y_test.argmax(axis=1) 
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)
 
#DNN
# loss :  0.14203384518623352
# acc :  0.9648148148148148

#LSTM
# loss :  0.9172805547714233
# acc :  0.6759259259259259

#Conv1D
# loss :  0.09772270917892456
# acc :  0.9685185185185186