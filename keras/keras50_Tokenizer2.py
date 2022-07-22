from keras.preprocessing.text import Tokenizer

text1 = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
text2 = '나는 지구용사 이재근이다. 멋있다. 또 또 얘기해봐'
# 전체데이터에서 활용도가 높은 객체가 앞 순서로 감.

token = Tokenizer()
token.fit_on_texts([text1,text2])

print(token.word_index) 

x = token.texts_to_sequences([text1,text2])
print(x) 

# one hot 을 해준다. 
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import numpy as np 

x_new = x[0] +x[1] #[2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9, 2, 10, 11, 12, 4, 4, 13]
print(x_new)

# x_new = to_categorical(x_new)
# print(x_new)
# print(x_new.shape) #(18, 14)

# #사이킷런
one = OneHotEncoder(categories='auto',sparse= False)#False로 할 경우 넘파이 배열로 반환된다.
x_new = np.array(x_new)
# print(x_new.shape)
x_new = x_new.reshape(-1,1)
one.fit(x_new)
x_new = one.transform(x_new)
print(x_new)
# print(x_new.shape)

