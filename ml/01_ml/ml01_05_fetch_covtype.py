import numpy as np 
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Input,Dropout,LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import datetime
import pandas as pd

#1.데이터 
datasets = fetch_covtype()
x= datasets.data
y= datasets.target


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=True,
                                                     random_state=58)

from sklearn.svm import LinearSVC
# #2.모델
model = LinearSVC()


#3.컴파일,훈련

model.fit(x_train,y_train)

result = model.score(x_test, y_test)

print('결과 acc : ', result)
y_predict = model.predict(x_test) 

# 결과 acc :  0.5366371397099321

