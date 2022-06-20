#1. 데이터
import numpy as np 
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim=1))  #input dim - input 레이어에 들어가는 데이터의 형태
model.add(Dense(50))               #input이 없는 이유 Sequential 순차적 모델이기 때문
model.add(Dense(30))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #mse- mean(평균),squad(제곱) (오차) 음수가 상쇄되는 경우가 있기때문 , optimizer(최적화)
model.fit(x, y, epochs=1000) #epochs-훈련횟수

#4. 평가, 예측
loss = model.evaluate(x, y)  # x,y 를 평가하다 
print('loss : ', loss) # loss 로 출력한다

result = model.predict([4])
print('4의 예측값 : ', result)

#레이어 층의 수 , 노드의 수 , 훈련랑 조절로 예측값 조정할 수 있다 (hyper parameter tuning) , 늘린다고 무조건 좋아지지 않는다.

# loss :  9.473903425812318e-15
# 4의 예측값 :  [[3.9999998]]
