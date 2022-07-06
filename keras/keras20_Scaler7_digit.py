import numpy as np 
from sklearn.datasets import load_digits
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,StandardScaler

#1.데이터 
datasets = load_digits()
x= datasets.data
y= datasets.target

print(x.shape, y.shape) #(1797, 64) (1797, ) * 64개의 컬럼이 아니라 8x8 이미지 (64칸) 그게 1797개가 있다.
print(np.unique(y,return_counts=True)) #y의 라벨 ,[0 1 2 3 4 5 6 7 8 9]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=True,
                                                     random_state=58)

# minmax , standard
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 

#2.모델
model = Sequential()
model.add(Dense(40, input_dim=64))
model.add(Dense(40,activation ='relu'))
model.add(Dense(40,activation ='relu'))
model.add(Dense(40,activation ='relu'))
model.add(Dense(30,activation ='relu'))
model.add(Dense(10,activation ='softmax')) #소프트맥스는 모든 연산값의 합이 1.0,그중 가장 큰값(퍼센트)을 선택,so 마지막 노드3개* y의 라벨의 갯수
#softmax는 아웃풋만 가능 히든에서 x

#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer ='adam', metrics='accuracy') #다중분류는 무조건 loss에 categorical_crossentropy
 #분류모델에서 셔플 중요! ,false로 하면 순차적으로 나와서 2가 아예 안나옴.

earlyStopping= EarlyStopping(monitor='val_loss',patience=30,mode='min',restore_best_weights=True,verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=32,validation_split=0.2,callbacks=earlyStopping, verbose=1)

#4.평가,예측
# loss,acc = model.evaluate(x_test,y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)
#################### 위와 동일###############
results = model.evaluate(x_test,y_test)
print('loss : ', results[0])
# print('accuracy : ', results[1])
############################################

# print(y_test)
y_predict = model.predict(x_test) 
y_predict = y_predict.argmax(axis=1) 


y_test = y_test.argmax(axis=1) 
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)

print(y_predict)
print(y_test)

# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[0])
# plt.show()

#minmax
# loss :  0.21550600230693817
# acc :  0.9425925925925925

#standard
# loss :  0.18987935781478882
# acc :  0.9518518518518518

#none
# loss :  0.28474682569503784
# acc :  0.9481481481481482

