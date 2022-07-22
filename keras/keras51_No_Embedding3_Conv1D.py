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
print(token.word_index)
# {'참': 1, '너무': 2, '재밋어요': 3, '최고에요': 4, '잘': 5, '만든': 6, 
#  '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, 
#  '더': 13, '보고': 14, '싶네요': 15, '글세요': 16, '별로
# 에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, 
# '재미없어요': 22, '재미없다': 23, '재밋네요': 24, '민수가': 25, '못': 26, 
# '생기긴': 27, '했어요': 28, '안결': 29, '혼해요': 30}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], 
#  [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27, 28], [29, 30]]

from keras.preprocessing.sequence import pad_sequences 
pad_x = pad_sequences(x, padding='pre', maxlen=5)  # 통상적으로 앞부분에 0을 채워주게 됨. padding = pre : 앞에 채우겟다. maxlen : 최대를 5개까지로 한정한다.
print(pad_x)
print(pad_x.shape) #(14, 5)
pad_x =pad_x.reshape(14,5,1)
word_size = len(token.word_index)   
print('word_size : ',word_size)     #30    단어사전의 갯수:30

print(np.unique(pad_x, return_counts =True))

#x = (14,5) y, (14, )

#2.모델 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LSTM,Embedding,Conv1D # 엠베딩은 통상 input layer쪽에서 많이 씀.

model = Sequential()            #인풋은 (14,5)
                    #input단어사전의 갯수, output 아웃풋갯수, input_length= 명시하지 않아도 잡아줌. 엠베딩layer 연산량은 동일
# model.add(Embedding(input_dim =31,output_dim=11, input_length=5))
# model.add(Embedding(input_dim =31,output_dim=10))
# model.add(Embedding(31,10))# 파라미터 다 안써도 가능 알지?
# model.add(Embedding(31,10,5))#error  *input_length는 안쓰거나 정확한 파라미터를 넣어달라!
# model.add(Embedding(31, 3, input_length=5))
# model.add(LSTM(32))
model.add(Conv1D(32, 2,input_shape =(5,1),activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3.컴파일,훈련 
model.compile(loss= 'binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(pad_x, labels, epochs=20, batch_size= 16)

#4.평가 ,예측
acc = model.evaluate(pad_x, labels)[1] #0으로 하면 loss 값 반환 
print('acc : ', acc)

