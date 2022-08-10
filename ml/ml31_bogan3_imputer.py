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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer # KNN: 비지도학습의 클러스터방식의 학습 


# imputer = SimpleImputer()#defalt는 mean
# imputer = SimpleImputer(strategy='mean')
# imputer = SimpleImputer(strategy='median')
# imputer = SimpleImputer(strategy='most_frequent')# 가장빈번하게 쓰는것을 채운다,동일할때는 앞에것을 씀 
# imputer = SimpleImputer(strategy='constant')#상수를 넣는데 defalt가 0
# imputer = SimpleImputer(strategy='constant', fill_value=777)
# imputer = KNNImputer() # defalt는 mean 
imputer = IterativeImputer()


imputer.fit(data)
data2 = imputer.transform(data)

print(data2)



