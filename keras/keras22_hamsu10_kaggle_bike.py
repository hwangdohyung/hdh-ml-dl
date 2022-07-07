import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import datetime as dt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv') 
            
test_set = pd.read_csv(path + 'test.csv') 


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

print(train_set)
print(test_set)


x = train_set.drop(['count'], axis=1)  
print(x)
print(x.columns)
print(x.shape) # (10886, 12)

y = train_set['count'] 
print(y)
print(y.shape) # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,random_state=31)   
      
# minmax , standard
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 
test_set = scaler.transform(test_set)# **최종테스트셋이 있는경우 여기도 스케일링을 적용해야함 **               
                                                    

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=12))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=31)      
from tensorflow.python.keras.callbacks  import EarlyStopping                                      
earlyStopping =EarlyStopping(monitor = 'val_loss',patience=30,mode='min',restore_best_weights=True,verbose=1)
model.fit(x_train, y_train, epochs=800, batch_size=60,validation_split=0.2,callbacks=earlyStopping, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x, y) 
print('loss : ', loss)

y_predict = model.predict(x_test)


# y_summit = model.predict(test_set)

# print(y_summit)
# print(y_summit.shape) # (6493, 1)

# submission_set = pd.read_csv(path + 'sampleSubmission.csv',index_col=0) 
# print(submission_set)

# y_summit = submission_set['count'] 

# abs(submission_set)
# print(submission_set)


# submission_set.to_csv(path + 'samplesubmission.csv', index = True)

def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


# loss :  1380.9127197265625
# RMSE :  48.02617663859703