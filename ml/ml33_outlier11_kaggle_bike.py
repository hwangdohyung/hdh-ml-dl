import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input,Dropout,LSTM
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import datetime as dt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from tensorflow.python.keras.callbacks  import EarlyStopping,ModelCheckpoint
import datetime
     
#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv') 
            
test_set = pd.read_csv(path + 'test.csv') 

###########이상치 처리##############
def dr_outlier(train_set):
    quartile_1 = train_set.quantile(0.25)
    quartile_3 = train_set.quantile(0.75)
    IQR = quartile_3 - quartile_1
    condition = (train_set < (quartile_1 - 1.5 * IQR)) | (train_set > (quartile_3 + 1.5 * IQR))
    condition = condition.any(axis=1)
    search_df = train_set[condition]

    return train_set, train_set.drop(train_set.index, axis=0)

dr_outlier(train_set)

###############################
train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) 
train_set.drop('casual',axis=1,inplace=True)
train_set.drop('registered',axis=1,inplace=True)

test_set.drop('datetime',axis=1,inplace=True) 


x = train_set.drop(['count'], axis=1)  
print(x)
print(x.columns)
print(x.shape) # (10886, 12)

y = train_set['count'] 
print(y)
print(y.shape) # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,random_state=31)   
      

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 
test_set = scaler.transform(test_set)      

print(x_train.shape,x_test.shape)



#2.모델구성
model = Sequential()
model.add(Dense(200,activation='relu', input_dim= 12)) 
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
    
earlyStopping =EarlyStopping(monitor = 'val_loss',patience=10,mode='min',restore_best_weights=True,verbose=1)

# filepath='./_ModelCheckpoint/k24/'
# filename='{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor = 'val_loss',mode = 'auto', save_best_only=True, verbose=1,
#                       filepath="".join([filepath,'kaggle_bike',date,'_',filename]))

model.fit(x_train, y_train, epochs=200, batch_size=64,validation_split=0.2,callbacks=[earlyStopping], verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test)


def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


### 이상치 처리 전 ###
# loss :  1842.0716552734375
# RMSE :  42.91936556947653

### 이상치 처리 후 ###

# loss :  1644.31591796875
# RMSE :  40.55016420942538