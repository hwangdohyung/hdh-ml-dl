from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score,accuracy_score
# predict 파일도 스케일링 하는거 주의! test파일 따로 있을때!
# 학습 데이터에 fit 한 설정을 그대로 test set에도 적용하는 것이다.
# 이때 주의할 점은 test set에는 fit이나 fit_transform 메서드를 절대 쓰면 안된다는거!
#  만약 test set에도 fit을 해버리면 sclaer가 기존에 학습 데이터에 fit한 기준을 다 무시하고 테스트 데이터에 새로운 mean, variance값을 얻으면서 테스트 데이터까지 학습해버린다. 
# 테스트 데이터는 검증을 위해 남겨둔 셋이기 때문에 반드시 주의해야 한다.

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape)

# print(np.min(x)) #0.0  
# print(np.max(x)) #711.0 # 전처리는 컬럼별로 해야됨. 0~711 이렇게 하면 잘못된거임 , 이것은 이해를 돕기위한 공식 

# x = (x -np.min(x)) / (np.max(x) - np.min(x)) # 최소값과 최대값 범위를 나눠주는것 

x_train,x_test, y_train,y_test = train_test_split(x,y, test_size=0.3, random_state= 66)

# 트레인테스트 split (범위 문제*피처스케일링) 트레인데이터 따로 스케일링해서 훈련시킨후 범위 밖의 테스트데이터에 대입해서 예측한다.val도 마찬가지 
# train set fit 하고 나머지데이터는 후에 transform

# minmax , standard ,maxabs , robust
# # scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 

# print(np.min(x_train)) #0.0
# print(np.max(x_train)) #1.0
# print(np.min(x_test)) #-0.06141956477526944
# print(np.max(x_test)) #1.1478180091225068

#2.모델구성
model = Sequential()
model.add(Dense(40,input_dim=13))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1,))

#3.컴파일,훈련

model.compile(loss='mse',optimizer='adam',)

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor= 'val_loss',patience=50,mode='min',restore_best_weights=True,verbose=1)
model.fit(x_train,y_train,epochs=1000,batch_size=10,verbose=1,validation_split= 0.2,callbacks=earlyStopping)

#4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 

# minmax
# loss: 9.25155353546142
# r2스코어 : 0.8880189163195478

# standard
# loss: 11.65905475616455
# r2스코어 : 0.8588784475617859

# maxabs
# loss:  11.972809791564941
# r2스코어 :  0.8550807368815204

#robust
# loss:  11.370014190673828
# r2스코어 :  0.862376994029175

# none
# loss: 14.690462112426758
# r2스코어 : 0.8221861795427536





