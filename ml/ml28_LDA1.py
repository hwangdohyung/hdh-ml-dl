from unittest import result
import numpy as np 
import pandas as pd 
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#1.데이터 
datasets = fetch_covtype()
x = datasets.data
y = datasets.target 
print(x.shape)   #(581012, 54)

le = LabelEncoder()
y = le.fit_transform(y)

# pca = PCA(n_components=20)
# x= pca.fit_transform(x) 

lda = LinearDiscriminantAnalysis(n_components=6) # y의 라벨 갯수(7개-1)보다 크면 안된다.
lda.fit(x, y)
x = lda.transform(x)


# pca_EVR = pca.explained_variance_ratio_ 
# cumsum = np.cumsum(pca_EVR)
# print(cumsum)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=123, shuffle=True, stratify=y
)


print(np.unique(y_train, return_counts= True))  # array([1, 2, 3, 4, 5, 6, 7]

#2.모델 
from xgboost import XGBClassifier
model = XGBClassifier(tree_method = 'gpu_hist',predictor = 'gpu_predictor', gpu_id=0,)


#3.훈련
import time
start = time.time() 
model.fit(x_train, y_train)
end = time.time()


#4.평가,예측
results = model.score(x_test,y_test)
print('결과 : ', results)
print('걸린시간 : ', round(end-start,2))

# 결과 :  0.86
# 걸린시간 :  91.93

# gpu 사용 
# 결과 :  0.86
# 걸린시간 :  6.2

# pca gpu/n_component : 10 
# 결과 :  0.84
# 걸린시간 :  4.26

# pca gpu/n_component : 20
# 결과 :  0.88
# 걸린시간 :  4.73

# lda gpu/n_component : 5
# 결과 :  0.7707460220476235
# 걸린시간 :  3.51

