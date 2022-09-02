import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1.데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]])
                #2개의 feature,컬럼,열     
y = np.array([11,12,13,14,15,16,17,18,19,20]) #y=wx+b w=1 b=10
print(x.shape) # (2,10)
print(y.shape) # (10,)

#넘파이 행렬 치환
x = x.T # x값에 덮어쓰기가 됨.
print(x.shape)
# x= x.transpose 

#resahpe 데이터 모양만 바뀜. 순서가 안바뀜. 지금 의도와 안맞음. 평소때는 훨씬 많이 씀.


#2.모델구성
model = Sequential()
model.add(Dense(50, input_dim=2)) # 특성2개 행무시 열우선
model.add(Dense(40, )) 
model.add(Dense(30, )) 
model.add(Dense(40, )) 
model.add(Dense(40, )) 
model.add(Dense(10, ))
model.add(Dense(1, )) 
 

#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)

#4.평가,예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([[10, 1.4]]) # 예측값도 feature 형태 동일시 해줘야함!
print('[10, 1.4]의 예측값 : ', result)
# loss :  2.1036248298855753e-08
# [10, 1.4]의 예측값 :  [[20.000286]]


#데이터의 유형
# 선형대수에서 다루는 데이터는 개수나 형태에 따라 크게 스칼라(scalar), 벡터(vector), 행렬(matrix), 텐서(tensor) 유형으로 나뉜다
# 스칼라(0차원): 숫자 하나로 이루어진 데이터 x =[1,2,3]에서 1,2,3 하나를 말함 스칼라 3개 
# 벡터(1차원): 스칼라의 모임 x =[1,2,3] 벡터1개 shape=(3,) 
# 행렬(matrix,2차원):벡터,즉 데이터 레코드가 여러인 데이터 집합 ([1,2,3],[4,3,2]) shape =(2,3)
# 텐서(3차원):같은 크기의 행렬이 여러 개 있는 것([[1,2,3],[4,3,2]] , [[4,3,11],[3,7,16]]) shape =(2,2,3) 가장 작은것부터 읽는다 (행,렬)
# tensorflow = 텐서를 연산시키다 피쳐의 숫자는 동일하다.

#외우기! 1.행무시 열우선 2.2개이상은 리스트

# #문제 1.[[1,2],[3,4],[5,6]]           -(3,2)
#       2.[[[1,2,3,4,5]]]               -(1,1,5)
#       3.[[1,2,3],[1,2,3],[4,5,6]]     -(3,3)
#       4.[4,3,2,1]                     -(4,)     *행렬이 아니라 4개의 스칼라 1개의 벡터 그래서 이렇게 표기
#       5.[[[[1,2,3],[4,5,6]]]]         -(1,1,2,3)