from keras.datasets import imdb
import numpy as np 

(x_train, y_train) , (x_test, y_test) = imdb.load_data(num_words = 1000)
print(y_train)
print(np.unique(y_train,return_counts =True))
print(len(np.unique(y_train)))     #2                
print(len(x_train[0]))   #218          
print(len(x_train[1]))   #189      
 
print('최대길이 : ', max(len(i) for i in x_train))        #2494    
print('평균길이 : ', sum(map(len,x_train))/ len(x_train)) #238.71


#전처리 
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
                       
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print('#######################################')
print(x_train.shape)             
print(y_train.shape)                  
print(x_test.shape)                       
print(y_test.shape)                     


#2.모델
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Conv1D,Flatten,Conv2D,Input,Embedding

model = Sequential()        
model.add(Embedding(input_dim =1000,output_dim=11,input_length=100))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

#3.컴파일,훈련 
model.compile(loss= 'categorical_crossentropy',optimizer='adam',metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=20, batch_size= 64)

#4.평가
accuracy = hist.history['acc']
print('accuracy: ', accuracy[-1])
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[-1])

