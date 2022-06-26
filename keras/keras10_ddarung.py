# 데이콘 따릉이 문제풀이 
import numpy as np 
import pandas as pd # csv 파일 당겨올 때 사용
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 

#1.데이터 
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col=0) #컬럼중에 id컬럼(0번째)은 단순 index 

print(train_set)
print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path + 'test.csv', index_col=0)  #예측에서 쓴다!

print(test_set)
print(test_set.shape) # (715, 9)

print(train_set.columns)
print(train_set.info()) #non-null count : 결측치 
print(train_set.describe()) 

#####결측치 처리 1. 제거 ######
print(train_set.isnull().sum()) 
train_set= train_set.fillna(train_set.mean())
print(train_set.isnull().sum()) 
print(train_set.shape)
##############################



x = train_set.drop(['count'], axis=1,)

# print(x)
# print(x.columns)
# print(x.shape) # (1459, 9)

y = train_set['count']
# print(y)
# print(y.shape) #(1459, )
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=71)
#2.모델구성
model = Sequential()
model.add(Dense(6, input_dim=9))
model.add(Dense(10))
model.add(Dense(2))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer ='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=25,verbose=2)


#4.평가,예측
loss = model.evaluate(x, y)
print('loss: ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt: 루트 씌우기

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)



#loss nan이 뜨는 이유 : 데이터에 nan값이 있기때문에 --해결법 기초 nan값을 지워준다.(결측치 처리)

# loss:  2985.57861328125
# RMSE :  56.34142663157544