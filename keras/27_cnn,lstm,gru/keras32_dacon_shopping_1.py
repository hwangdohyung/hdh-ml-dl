
import numpy as np 
import pandas as pd # csv 파일 당겨올 때 사용
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
from collections import Counter
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

#1.데이터 
path = './_data/shopping/'
train_set = pd.read_csv(path + 'train.csv', index_col=0) #컬럼중에 id컬럼(0번째)은 단순 index 
print(train_set)
print(train_set.shape) # (6255, 12)
test_set = pd.read_csv(path + 'test.csv', index_col=0)  #예측에서 쓴다!

#Date
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.Date)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.Date)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.Date)]
train_set['year'] = train_set['year'].map({2010:0, 2011:1, 2012:2})

test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.Date)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.Date)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.Date)]
test_set['year'] = test_set['year'].map({2010:0, 2011:1 , 2012:2})

#IsHoliday
train_set['IsHoliday'] = train_set['IsHoliday'].astype(int)
test_set['IsHoliday'] = test_set['IsHoliday'].astype(int)

# print(train_set)
# print(test_set)


print(train_set.info())

train_set = train_set.fillna(0)
test_set = test_set.fillna(0)

# 전처리 하기 전 칼럼들을 제거합니다.
train_set = train_set.drop(columns=['Date'])
test_set = test_set.drop(columns=['Date'])

# 학습에 사용할 정보와 예측하고자 하는 정보를 분리합니다.
x_train = train_set.drop(columns=['Weekly_Sales'])
y_train = train_set[['Weekly_Sales']]


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=48)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=13))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer ='adam')

earlyStopping =EarlyStopping(monitor = 'val_loss',patience=20,mode='min',restore_best_weights=True,verbose=1)

model.fit(x_train, y_train, epochs=3000, batch_size=100,validation_split=0.2,callbacks=[earlyStopping],verbose=2)

#4.평가,예측
loss = model.evaluate(x_train, y_train)
print('loss: ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_submmit = model.predict(test_set)
print(y_submmit)
print(y_submmit.shape)  

submission = pd.read_csv(path + 'sample_submission.csv')
submission['Weekly_Sales'] = y_submmit

submission.to_csv(path + 'submission.csv',index=False)

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

