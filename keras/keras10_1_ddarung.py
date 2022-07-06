# 데이콘 따릉이 문제풀이 
import numpy as np 
import pandas as pd # csv 파일 당겨올 때 사용
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
from collections import Counter

 
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

#####이상치 처리##############
def dr_outlier(train_set):
    quartile_1 = train_set.quantile(0.25)
    quartile_3 = train_set.quantile(0.75)
    IQR = quartile_3 - quartile_1
    condition = (train_set < (quartile_1 - 1.5 * IQR)) | (train_set > (quartile_3 + 1.5 * IQR))
    condition = condition.any(axis=1)
    search_df = train_set[condition]

    return train_set, train_set.drop(train_set.index, axis=0)


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
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=48)

# import matplotlib as mpl 
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm


#2.모델구성
model = Sequential()
model.add(Dense(100, input_dim=9))
model.add(Dense(100,activation ='selu'))
model.add(Dense(100,activation ='selu'))
model.add(Dense(100,activation ='selu'))
model.add(Dense(100,activation ='selu'))
model.add(Dense(100,activation ='selu'))
model.add(Dense(100,activation ='selu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer ='adam')
model.fit(x_train, y_train, epochs=3000, batch_size=150,verbose=2)


#4.평가,예측
loss = model.evaluate(x, y)
print('loss: ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt: 루트 씌우기

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_submmit = model.predict(test_set)
print(y_submmit)
print(y_submmit.shape)  #(715,1)

submission = pd.read_csv(path + 'samplesubmission.csv')
submission['count'] = y_submmit

submission.to_csv(path + 'samplesubmission.csv',index=False)

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

################.to_csv() 를 사용해서 
### submission.csv를 완성하시오!!!



#loss nan이 뜨는 이유 : 데이터에 nan값이 있기때문에 --해결법 기초 nan값을 지워준다.(결측치 처리)


# loss:  508.2084045410156
# RMSE :  39.72508807596856