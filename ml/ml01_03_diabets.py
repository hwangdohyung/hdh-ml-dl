#다중분류 point) --- loss categorical!, softmax ,마지막 노드갯수!,one hot encoding
import numpy as np 
from sklearn.datasets import load_diabetes
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC,SVC


import tensorflow as tf
tf.random.set_seed(66)

#1.데이터
datasets = load_diabetes()
x= datasets['data']
y= datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=True,
                                                     random_state=58) 

# #2.모델
model = SVC()


#3.컴파일,훈련

model.fit(x_train,y_train)

result = model.score(x_test, y_test)

print('결과 r2 : ', result)

