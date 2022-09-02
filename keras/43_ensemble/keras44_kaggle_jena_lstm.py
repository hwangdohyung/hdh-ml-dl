# evaluate까지만 lstm,conv1 사용 
import pandas as pd 
import numpy as np 

#1.데이터
path = './_data/kaggle_jena/'
data = pd.read_csv(path + 'jena_climate.csv') 

print(data.shape)
data = data.drop('Date Time',axis =1)

print(data.shape)
x_predict = data[]

print(x_predict)

