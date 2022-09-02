from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans 
import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score

datasets = load_breast_cancer()

x = pd.DataFrame(datasets.data, columns=[datasets.feature_names])

kmeans = KMeans(n_clusters=2, random_state=123) #n_clusters : 라벨의 갯수
kmeans.fit(x)

print(kmeans.labels_)

print(datasets.target)

x['cluster'] = kmeans.labels_
x['target'] = datasets.target

print(x)

print(accuracy_score(x['cluster'],x['target']))

# 0.14586994727592267

