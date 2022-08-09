import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk 
print(sk.__version__)
import warnings
warnings.filterwarnings(action='ignore')


#1.데이터
datasets = load_breast_cancer()
x = datasets.data 
y = datasets.target
print(x.shape,y.shape)     

pca = PCA(n_components=30)   
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


x_train,x_test,y_train,y_test = train_test_split(x,y , train_size=0.8, random_state=123, shuffle=True)

#2.모델
from sklearn.ensemble import RandomForestRegressor 
from xgboost import XGBRegressor

model = RandomForestRegressor()

#3.훈련 
model.fit(x_train, y_train) #, eval_metric= 'error')

#4.평가, 예측 
results = model.score(x_test, y_test)
print('결과 : ', results)



