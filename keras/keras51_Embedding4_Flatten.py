from keras.preprocessing.text import Tokenizer
import numpy as np 

#1.데이터
docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요','글세요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밋네요', '민수가 못 생기긴 했어요',
        '안결 혼해요']

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])      #(14, )

token = Tokenizer()
token.fit_on_texts(docs)
x = token.texts_to_sequences(docs)

from keras.preprocessing.sequence import pad_sequences 
pad_x = pad_sequences(x, padding='pre', maxlen=5 ) # truncating=  
print(pad_x)
print(pad_x.shape) #(14, 5)

word_size = len(token.word_index)   
print('word_size : ',word_size)     

#2.모델 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LSTM,Embedding,Flatten

model = Sequential()        
model.add(Embedding(input_dim =31,output_dim=11,input_length=5)) #flatten 해주려면 none값이 없어야함 input length 필수
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3.컴파일,훈련 
model.compile(loss= 'binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(pad_x, labels, epochs=20, batch_size= 16)

#4.평가 ,예측
# acc = model.evaluate()[1]
# print('acc : ', acc)

#### [실습] ########

x_predict = '나는 형권이가 정말 재미없다 너무 정말 '
# 결과는??? 긍정??? 부정???

token.fit_on_texts(x_predict)
x_predict = token.texts_to_sequences(x_predict)
x_predict= pad_sequences(x_predict, padding='pre', maxlen=5 )
result = model.predict(x_predict)[0]
print(result)
print('result : ', result)

if result >= 0.5 :
    print("긍정")
else :
    print("부정")
    