import numpy as np 
import pandas as pd # csv 파일 당겨올 때 사용
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
from collections import Counter
import math
 
#1.데이터 
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv', index_col=0) #컬럼중에 id컬럼(0번째)은 단순 index 

print(train_set)
print(train_set.shape)

test_set = pd.read_csv(path + 'test.csv', index_col=0)  #예측에서 쓴다!

from sklearn.preprocessing import MinMaxScaler

# # 객체 생성
# minmax_scaler = MinMaxScaler()

# # 분포 저장
# minmax_scaler.fit(train_set)

# # 스케일링
# minmax_scaled = minmax_scaler.transform(train_set)


# print(test_set)
# print(test_set.shape) 

# print(train_set.columns)
# print(train_set.info()) #non-null count : 결측치 
# print(train_set.describe()) 

# def dr_outlier(train_set):
#     quartile_1 = train_set.quantile(0.25)
#     quartile_3 = train_set.quantile(0.75)
#     IQR = quartile_3 - quartile_1
#     condition = (train_set < (quartile_1 - 1.5 * IQR)) | (train_set > (quartile_3 + 1.5 * IQR))
#     condition = condition.any(axis=1)
#     search_df = train_set[condition]

#     return train_set, train_set.drop(train_set.index, axis=0)




print(train_set)

train_set['temp'] = train_set['temp'].astype(int)
train_set['atemp'] = train_set['atemp'].astype(int)
train_set['windspeed'] = train_set['windspeed'].astype(int)

x = train_set.drop(['count','casual','registered'], axis=1,)

print(x)
print(x.columns)
print(x.shape) 
print(train_set.info())

y = train_set['count']
print(y)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=22)


#2.모델구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(100,activation ='relu'))
model.add(Dense(50,activation ='relu'))
model.add(Dense(50,activation ='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer ='adam')
model.fit(x_train, y_train, epochs=2000, batch_size=200,verbose=2)


#4.평가,예측
loss = model.evaluate(x, y)
print('loss: ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt: 루트 씌우기

y_submmit = model.predict(test_set)

print(y_submmit)
print(y_submmit.shape)  

submission = pd.read_csv(path + 'samplesubmission.csv')
submission['count'] = y_submmit

submission.to_csv(path + 'samplesubmission.csv',index=False)

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


#회귀모델:수치로 떨어지는것,#분류모델:ex)남자인지 여자인지 구별하는것
#2가지 뿐이다!