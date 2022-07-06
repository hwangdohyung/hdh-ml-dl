import numpy as np 
from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
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
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 

#2.모델
model = Sequential()
model.add(Dense(40, input_dim=13))
model.add(Dense(40,activation ='relu'))
model.add(Dense(40,activation ='relu'))
model.add(Dense(40,activation ='relu'))
model.add(Dense(40,activation ='relu'))
model.add(Dense(3,activation ='softmax'))


#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer ='adam', metrics='accuracy') 
 

earlyStopping= EarlyStopping(monitor='val_loss',patience=30,mode='min',restore_best_weights=True,verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=32,validation_split=0.2,callbacks=earlyStopping, verbose=1)

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

print(y_predict)
print(y_test)

#minmax
# loss :  0.959882378578186
# acc :  0.9259259259259259

#standard
# loss :  0.0613056980073452
# acc :  0.9629629629629629

#maxabs
# loss :  0.4688432514667511
# acc :  0.7962962962962963

#robust
# loss :  0.12918157875537872
# acc :  0.9629629629629629

#none
# loss :  0.3786742687225342
# acc :  0.8333333333333334