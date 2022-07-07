import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam') 
model.fit(x, y, epochs=100, batch_size=1)# batch = 데이터 하나씩 따로 작업하는것 *데이터가 많아질 때 overflow를 방지,batch size가 줄면 
#훈련 횟수 많아짐 단점은 시간이 오래걸림
#append= 하나씩 나온 결과를 연결하는 것 
# #iteration의 의미
# 마지막으로 iteration은 1-epoch를 마치는데 필요한 미니배치 갯수를 의미합니다. 
# 다른 말로, 1-epoch를 마치는데 필요한 파라미터 업데이트 횟수 이기도 합니다. 
# 각 미니 배치 마다 파라미터 업데이터가 한번씩 진행되므로 iteration은 
# 파라미터 업데이트 횟수이자 미니배치 갯수입니다. 
# 예를 들어, 700개의 데이터를 100개씩 7개의 미니배치로 나누었을때, 
# 1-epoch를 위해서는 7-iteration이 필요하며 7번의 파라미터 업데이트가 진행됩니다

#4. 평가, 예측
loss = model.evaluate(x, y)  
print('loss : ', loss)

result = model.predict([6])
print('6의 예측값 : ', result)

# loss :  0.4547010362148285
# 6의 예측값 :  [[5.957699]]

#y = w1*x1+w2*x2+w3*x3 1.금리 2.환율 3. 부동산     ex) x = ([1,2,3],[4,3,2],[4,5,6])
#과제 : 파이썬- 리스트에 대해 조사 메일로 보내기 ,행렬 읽는 문제 해보기, 