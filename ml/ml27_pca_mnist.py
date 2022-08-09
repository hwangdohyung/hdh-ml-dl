import numpy as np 
from sklearn.decomposition import PCA
from keras.datasets import mnist

(x_train, _),(x_test, _) = mnist.load_data()

print(x_train.shape, x_test.shape)
#(60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)

############################################################
# [실습]
# pca를 통해 0.95 dltkdls n_components는 몇개?
# 0.95
# 0.999
# 1.0
# 힌트 np.argmax
############################################################

x = x.reshape(70000,784)
print(x.shape) #(70000, 784)

pca = PCA(n_components=784)   
x= pca.fit_transform(x) 
print(x.shape)            

pca_EVR = pca.explained_variance_ratio_ # 새로 생성된 feature 들의 importance
print(pca_EVR)

print(sum(pca_EVR)) #0.999998352533973  *1이라고 볼수있다 

cumsum = np.cumsum(pca_EVR)
print(cumsum)

import matplotlib.pyplot as plt 
plt.plot(cumsum)
plt.grid()
plt.show()
print('=========================')

print(np.argmax(cumsum >= 0.95)+1)  # 154, 0.95가 되는 시작부분 
print(np.argmax(cumsum >= 0.99)+1)  # 331, 0.99가 되는 시작부분
print(np.argmax(cumsum >= 0.999)+1)  # 486, 0.999가 되는 시작부분
print(np.argmax(cumsum+1)+1)          # 713 , 1.0 가 되는 시작부분



