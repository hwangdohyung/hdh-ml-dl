import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(506, 13) (506, )
print(datasets.feature_names)
print(datasets.DESCR)


# #2.모델
# model = Sequential()
# model.add(Dense(100, input_dim=13)) 
# model.add(Dense(100,activation ='relu'))
# model.add(Dense(100,activation ='relu'))
# model.add(Dense(100,activation ='relu'))
# model.add(Dense(100,activation ='relu'))
# model.add(Dense(1))

# #3.컴파일,훈련
# x_train, x_test, y_train, y_test = train_test_split(x, y, 
#                                                     train_size=0.7,
#                                                     shuffle=True,
#                                                     random_state=66)
# model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train, epochs =1000, batch_size=100,
#           validation_split= 0.3)

# #4.평가,예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# y_predict = model.predict(x_test) 

# #R2결정계수(성능평가지표)
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict) 
# print('r2스코어 : ' , r2) 

# # loss :  18.26516342163086
# # r2스코어 :  0.7789178709122163

