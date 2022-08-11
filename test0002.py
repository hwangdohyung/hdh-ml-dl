# 데이콘 따릉이 문제풀이 
import numpy as np 
import pandas as pd # csv 파일 당겨올 때 사용
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Input,LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
from collections import Counter
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import datetime
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

#1.데이터 
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col=0) #컬럼중에 id컬럼(0번째)은 단순 index 

print(train_set)
print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path + 'test.csv', index_col=0)  #예측에서 쓴다!

print(train_set.isnull().sum())
print(train_set)

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

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=48)

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape,x_test.shape)


#2.모델구성
model = Sequential()
model.add(Dense(200,activation='relu', input_dim=9)) 
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer ='adam')

earlyStopping =EarlyStopping(monitor = 'val_loss',patience=10,mode='min',restore_best_weights=True,verbose=1)

# filepath = './_ModelCheckpoint/k24/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor= 'val_loss',mode = 'auto',save_best_only=True, verbose = 1,
#                       filepath = "".join([filepath,'ddarung',date,'_',filename]))

model.fit(x_train, y_train, epochs=200, batch_size=100,validation_split=0.2,callbacks=[earlyStopping],verbose=2)

#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

#이상치 처리 #
# loss:  1981.86376953125
# RMSE :  44.51813035128636
# 그냥 #
# loss:  2256.03076171875
# RMSE :  47.497690725850234