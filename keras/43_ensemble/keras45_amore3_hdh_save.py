import numpy as np 
import pandas as pd 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,GRU,LSTM,Flatten,Conv2D,Conv1D,MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping


#1.데이터
path = './_data/test_amore_0718/'
data1= pd.read_csv(path + '아모레220718.csv', thousands=',',encoding='euc-kr') 
data2 = pd.read_csv(path + '삼성전자220718.csv', thousands=',',encoding='euc-kr') 
data1 = data1.sort_values(by='일자', ascending=True) #오름차순 정렬
data2 = data2.sort_values(by='일자', ascending=True) #오름차순 정렬


data1 = data1[['시가','고가','저가','Unnamed: 6','거래량','금액(백만)','신용비','종가']]
data2 = data2[['시가','고가','저가','Unnamed: 6','거래량','금액(백만)','신용비','종가']]

print(data2)
print(data2.shape)

data1 = data1.drop(range(0,2180))
data2 = data2.drop(range(0,2039))

print(data1,data1.shape)
print(data2,data2.shape)

'''
data1 = data1.dropna(axis=0)
data2 = data2.dropna(axis=0)


data4 = data1.drop(['종가'],axis=1)
jong = data1['종가']


scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# scaler = RobustScaler()
# # scaler.fit(x_train)
# # x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
# # x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 

data4 = scaler.fit_transform(data4)
data2 = scaler.fit_transform(data2)

print(data1,data1.shape)

#데이터 자르기

def split_3(dataset,time_steps,y_column):
    xs=list()
    ys=list()
    for i in range(0,len(dataset)-time_steps+1):
        x=dataset[i:(i+time_steps),:-1]
        y=dataset[i+time_steps-1:(i+time_steps-1)+y_column,-1]

        xs.append(x)
        ys.append(y)
    return np.array(xs),np.array(ys)

x1,y1 = split_3(data1,5,1)
x2,y2 = split_3(data2,5,1)

print(x1.shape,y1.shape) #(1767,5,7) (1767,1)
print(x2.shape,y2.shape) #(1031,5,7) (1031,1)

#2.모델

model = Sequential()
model.add(LSTM(200,activation='relu', input_shape=(5,7))) 
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
earlyStopping = EarlyStopping(monitor= 'loss',patience=10, mode='min',restore_best_weights=True,verbose=1)
model.fit(x1,y1, epochs=1,batch_size=32,verbose=1,callbacks=earlyStopping)

#4.평가,예측 
loss = model.evaluate(x1,y1)

result = model.predict(x1,y1)  #모델은 3차원을 원한다. 
print('loss : ', loss)
print('내일 시가 : ', result[-1:])
# print('result : ', result)
'''