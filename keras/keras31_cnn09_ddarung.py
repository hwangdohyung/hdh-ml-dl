# 데이콘 따릉이 문제풀이 
import numpy as np 
import pandas as pd # csv 파일 당겨올 때 사용
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Input,Flatten,Conv2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
from collections import Counter
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import datetime

#1.데이터 
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col=0) #컬럼중에 id컬럼(0번째)은 단순 index 

print(train_set)
print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path + 'test.csv', index_col=0)  #예측에서 쓴다!

train_set = train_set.dropna()
test_set= test_set.fillna(test_set.mean())

x = train_set.drop(['count'], axis=1,)
y = train_set['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=48)

print(x_train.shape,x_test.shape)


scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(929,3,3,1)
x_test = x_test.reshape(399,3,3,1)
print(x_train.shape,x_test.shape)



#2.모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same', input_shape=(3,3,1))) 
model.add(Conv2D(7, (2,2), padding='same', activation='relu'))
model.add(Conv2D(7, (2,2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1)) 

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')


#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer ='adam')

earlyStopping =EarlyStopping(monitor = 'val_loss',patience=30,mode='min',restore_best_weights=True,verbose=1)

# filepath = './_ModelCheckpoint/k24/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor= 'val_loss',mode = 'auto',save_best_only=True, verbose = 1,
#                       filepath = "".join([filepath,'ddarung',date,'_',filename]))

model.fit(x_train, y_train, epochs=3000, batch_size=100,validation_split=0.2,callbacks=[earlyStopping],verbose=2)

#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
#DNN
# loss:  93992096.0
# RMSE :  43.050460645811704

#CNN
# loss:  2277.371337890625
# RMSE :  47.721810217450205
