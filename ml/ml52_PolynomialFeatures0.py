import numpy as np 
import pandas as pd 
from sklearn.preprocessing import PolynomialFeatures # 전처리 단계에서 증폭개념

x = np.arange(8).reshape(4, 2)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]

# pf = PolynomialFeatures(degree=2)
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]

# 맨앞은 1, 다음 자신(2,3), 2^, 2*3, 3^

pf = PolynomialFeatures(degree=3)

#  [[  1.   0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  1.   2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  1.   4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  1.   6.   7.  36.  42.  49. 216. 252. 294. 343.]]

x_pf = pf.fit_transform(x)

print(x_pf)
print(x_pf.shape)


