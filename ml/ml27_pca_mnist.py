import numpy as np 
from sklearn.decomposition import PCA
from keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

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

x_train = x_train.reshape(60000,784)

#2.모델
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from xgboost import XGBRegressor

model = RandomForestClassifier()

#3.훈련 
model.fit(x_train,y_train) #, eval_metric= 'error')

#4.평가, 예측 
results = model.score(x_test, y_test)
print('결과 : ', results)




