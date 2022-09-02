from keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index) #{'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}

x = token.texts_to_sequences([text])
print(x) #[[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]]

# one hot 을 해준다. 
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import numpy as np 
# x = to_categorical(x)
# print(x)
# print(x.shape) #(1,11,9)
# 자연어 처리는 기본적으로 시계열을 깔고 한다.(LSTM)

# #사이킷런
one = OneHotEncoder(categories='auto',sparse= False)#False로 할 경우 넘파이 배열로 반환된다.
x = np.array(x)
x = x.reshape(-1,1)
one.fit(x)
x = one.transform(x)
print(x)
print(x.shape) #(11,8)