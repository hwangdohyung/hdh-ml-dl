from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D,Flatten,MaxPool2D # 이미지 작업은 2차원

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3),padding ='same',input_shape = (28, 28, 1)))    
model.add(MaxPool2D())# 14,14, 64
model.add(Conv2D(32, (2,2),padding='valid',activation ='relu'))#padding의 디폴트값 - valid  #13,13,32
model.add(MaxPool2D())  #6,6,32
model.add(Flatten()) 
model.add(Dense(32,activation = 'relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))
model.summary() 

# 이미지 가장자리에 중요한 정보들이 모여 있다면 계속 압축할 경우,shape이 계속 작아지며, 소멸될 가능성이 있다. 
#패딩 -주위를 0으로 둘러싼다(제로패딩). 가장자리 정보손실을 막기위함 커널사이즈와 상관없이 원래 shape 이 나옴

#맥스풀링- 수영장에서 가장큰값만 뺀다,겹치는 부분없이 커널사이즈로 자른다. 자원손실 줄임.

