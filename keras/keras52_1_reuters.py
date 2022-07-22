from keras.datasets import reuters
import numpy as np 
import pandas as pd 

(x_train,y_train) , (x_test,y_test) = reuters.load_data(num_words=1000, test_split=0.2)

print(x_train)
print(x_train.shape,x_test.shape) #(8982, ) ,(2246, ) 리스트가 8982개란 뜻

print(y_train)
print(np.unique(y_train,return_counts =True))
print(len(np.unique(y_train)))     # 46
print(type(x_train),type(y_train)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0]))            #<class 'list'>
# print(x_train[0].shape)          #error 'list' object has no attribute 'shape' 리스트는 shape 안먹힘.
print(len(x_train[0]))             #87 len을 활용
print(len(x_train[1]))             #56 
 
print('뉴스기사의 최대길이 : ', max(len(i) for i in x_train))        # 2376 ->8982번 도는데 반환값이 앞에 들어가게됨 . 그것의 길이가 반환. 그것을 max해줌  
print('뉴스기사의 평균길이 : ', sum(map(len,x_train))/ len(x_train)) # 145.53


#전처리 
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
                        #(8982,) -> (8982,100)
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print('#######################################')
print(x_train.shape)  #(8982, 100)              
print(y_train.shape)  #(8982, 46)                 
print(x_test.shape)   #(2246, 100)                     
print(y_test.shape)   #(2246, 46)                     

#2.모델
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,LSTM,Input,Conv2D,Conv1D,Embedding,Flatten

model = Sequential()        
model.add(Embedding(input_dim =1000,output_dim=11,input_length=100))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(46, activation='softmax'))

model.summary()

#3.컴파일,훈련 
model.compile(loss= 'categorical_crossentropy',optimizer='adam',metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=20, batch_size= 64)

#4.평가
accuracy = hist.history['acc']
print('accuracy: ', accuracy[-1])
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[-1])
#accuracy:  0.9533511400222778
# loss :  0.641139805316925