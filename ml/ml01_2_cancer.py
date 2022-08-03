import numpy as np 
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVR

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=66)


model = LinearSVR()

model.fit(x_train,y_train)

#4.평가,예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)
y_predict= model.predict(x_test)
result = model.score(x_test,y_test) #훈련시키지 않은 부분을 평가 해야 되기 때문에 x_test.
print('결과 : ', result)


#R2결정계수(성능평가지표)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 


