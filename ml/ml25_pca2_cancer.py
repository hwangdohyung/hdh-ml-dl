import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk 
print(sk.__version__)
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
tf.random.set_seed(66)

#1.데이터
datasets = load_breast_cancer()
x = datasets.data 
y = datasets.target
print(x.shape,y.shape)      #(569, 30) (569,)

pca = PCA(n_components=4)  
x= pca.fit_transform(x) 
print(x.shape)              

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

# (569, 30) (569,)
# 결과 :  0.9099502839959906

# pca = PCA(n_components=4)  (569, 4)
# 결과 :  0.9066441697293685



