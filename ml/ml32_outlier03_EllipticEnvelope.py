import numpy as np 
aaa = np.array([-10,2,3,4,5,6,700,8,9,10,11,12,50])
aaa = aaa.reshape(-1, 1)
print(aaa.shape)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1) # .1의 의미 10프로를 넘어서는 범위를 이상치로 하겠다.

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)
#[ 1  1  1  1  1  1 -1  1  1  1  1  1 -1] : 이상치의 위치 