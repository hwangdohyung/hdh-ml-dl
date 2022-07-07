import numpy as np 
from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Input
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
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 

model=load_model('./_save/keras23_15_save_wine.h5')

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