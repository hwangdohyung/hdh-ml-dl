# 데이콘 따릉이 문제풀이 
import numpy as np 
import pandas as pd # csv 파일 당겨올 때 사용
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
from collections import Counter
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
 
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
# print(train_set.isnull().sum()) 
train_set = train_set.dropna()
# print(train_set.isnull().sum()) 
# print(train_set.shape)
test_set= test_set.fillna(test_set.mean())
##############################


x = train_set.drop(['count'], axis=1,)

# print(x)
# print(x.columns)
# print(x.shape) # (1459, 9)

y = train_set['count']
# print(y)
# print(y.shape) #(1459, )

# import matplotlib as mpl 
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=48)

# minmax , standard
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 
test_set = scaler.transform(test_set)# **최종테스트셋이 있는경우 여기도 스케일링을 적용해야함 **

model= load_model('./_save/keras23_21_save_ddarung.h5')

#4.평가,예측
loss = model.evaluate(x, y)
print('loss: ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt: 루트 씌우기


rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

