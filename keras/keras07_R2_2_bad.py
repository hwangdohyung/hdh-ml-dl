#1. r2를 음수가 아닌 05. 이하로 만들것
#2. 데이터 건들지 마 
#3. 레이어는 인풋 아웃풋 포함 7개 이상
#4. batch_size=1
#5. 히든레이어의 한 레이어당 노드는 10개 이상 100개 이하
#6. train 70% 
#7. epoch 100번 이상


import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])
 
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=66)    

#2.모델구성
model = Sequential()
model.add(Dense(90, input_dim=1))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)

#R2결정계수(성능평가지표)
from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print('r2스코어 : ' , r2)




# import matplotlib.pyplot as plt
# plt.scatter(x, y) 
# plt.plot(x, y_predict, color='orange') 
# plt.show() 

#나쁜모델 만들기(레이어 수와 노드 수 많이 늘리면 loss,r2 커짐)
# loss :  26.349031448364258
# r2스코어 :  0.025181422178507606