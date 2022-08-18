from sklearn.datasets import load_iris
from sklearn.cluster import KMeans # y라벨을 생성하는것 비지도 학습을 통해! 
import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score

datasets = load_iris()

x = pd.DataFrame(datasets.data, columns=[datasets.feature_names])

print(x) #(150, 4)

kmeans = KMeans(n_clusters=3, random_state=1234) #n_clusters : 라벨의 갯수
kmeans.fit(x)

print(kmeans.labels_)

print(datasets.target)

#[실습] accuracy_score 구해라 !!

x['cluster'] = kmeans.labels_
x['target'] = datasets.target

print(x)

print(accuracy_score(x['cluster'],x['target']))



# 0.8933333333333333

