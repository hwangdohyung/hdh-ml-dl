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
pad_x = pad_sequences(x, padding='pre', maxlen=5) # truncating=  
print(pad_x)
print(pad_x.shape) #(14, 5)

word_size = len(token.word_index)   
print('word_size : ',word_size)     

print(pad_x.shape)

#2.모델 
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,LSTM,Embedding,Input

input1 =Input(shape= (5,))        
embedding1 = Embedding(31,10, input_length=5)(input1)
lstm1 = LSTM(32)(embedding1)
output1 = Dense(1,activation='sigmoid')
model = Model(inputs= input1,outputs=output1)

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
x_predict= pad_sequences(x_predict, padding='pre', maxlen=6)
result = model.predict(x_predict)[0]
print(result)

if result >= 0.5 :
    print("긍정")
else :
    print("부정")
    
# [0.47885357]
# 부정    

