import numpy as np 
import matplotlib.pyplot as plt 
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])
aaa = np.transpose(aaa)

print(aaa.shape)
print(aaa)
aaa1 = aaa[:,0]
print(aaa1)
bbb = aaa[:,1]
print(bbb)

aaa1 = aaa1.reshape(-1,1)
print(aaa1)
bbb = bbb.reshape(-1,1)
print(bbb)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1) # .1의 의미 10프로를 넘어서는 범위를 이상치로 하겠다.
outliers.fit(aaa1)
outliers.fit(bbb)

results1 = outliers.predict(aaa1)
results2 = outliers.predict(bbb)

print(results1)
print(results2)

# [1 1 1 1 1 1 1 1 1 1 1 1 1]
# [ 1  1  1  1  1  1 -1  1  1 -1  1  1  1]