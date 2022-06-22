import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# #1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])         #30프로 중간중간 데이터를 빼주는 것이 몰아서 빼는것 보다 성능이 좋다
y = np.array([1,2,3,4,5,6,7,8,9,10])         #ex)x_train(1,2,4,5,6,8,9)
                                             #   x_test(3,7,10) 셔플을 사용하여 랜덤하게 빼준다.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    # train_size=0.7, 
                                                    # shuffle=True, 
                                                    random_state =66)
#random값이 바뀌면 데이터 바뀜

#[검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법 찾아라
print(x_train)  
print(x_test)    
print(y_train)   
print(y_test)


#2.모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse' , optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('11의 예측값 : ', result) 

# loss :  0.001300719566643238
# 11의 예측값 :  [[10.953842]]

