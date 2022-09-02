import numpy as np 
import pandas as pd 
from sklearn.preprocessing import PolynomialFeatures # 전처리 단계에서 증폭개념

x = np.arange(12).reshape(4, 3)
print(x)
print(x.shape)

# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]
# (4, 3)

# pf = PolynomialFeatures(degree=2)
# [[  1.   0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  1.   3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  1.   6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  1.   9.  10.  11.  81.  90.  99. 100. 110. 121.]]
# (4, 10)

pf = PolynomialFeatures(degree=3)


x_pf = pf.fit_transform(x)

print(x_pf)
print(x_pf.shape)







