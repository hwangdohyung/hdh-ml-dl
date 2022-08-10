import numpy as np 
import pandas as pd 

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])

data = data.transpose()
data.columns = ['x1','x2','x3','x4']
print(data)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

#결측치 확인 
print(data.isnull())
print(data.isnull().sum())
print(data.info())

#1.결측치 삭제 
print("===============결측치 삭제===============")
print(data.dropna())#defalt axis=0
print(data.dropna(axis=1))

#2-1. 특정값 - 평균
means = data.mean() #컬럼별 평균
print('평균 : ', means)
data2 = data.fillna(means)
print(data2)

#2-2. 특정값 - 중위값
median = data.median() 
print('중위값 : ', median)
data3 = data.fillna(median)
print(data3)

