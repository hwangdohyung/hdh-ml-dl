import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import pandas as pd


#1. 데이터
x = np.array([[1,2,3],[4,5,6],[7,8,9]])
y = np.array([1,2,3,4,5])

print(x)


x = x.drop(range(0,1),axis=0)
print(x)
