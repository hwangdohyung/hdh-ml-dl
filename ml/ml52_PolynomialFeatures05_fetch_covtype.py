from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline 
from sklearn.svm import LinearSVC
#1.데이터 
datasets = fetch_covtype()
x, y = datasets.data, datasets.target 
print(x.shape,y.shape) 

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True,random_state=1234,stratify=y)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

#2.모델 
model = make_pipeline(StandardScaler(),LinearSVC(),)
model.fit(x_train,y_train)
print('그냥: ' , model.score(x_test,y_test)) 

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring= 'accuracy')
print('그냥CV : ', scores)
print('그냥CV 엔빵 : ', np.mean(scores))


##################### polynomial 후 ########################

pf = PolynomialFeatures(degree=2, 
                        # include_bias=False
                        )

xp = pf.fit_transform(x)
print(xp.shape)             

x_train,x_test,y_train,y_test = train_test_split(xp,y, train_size=0.8, shuffle=True,random_state=1234,stratify=y)

#2.모델 
model = make_pipeline(StandardScaler(),LinearSVC(),)
model.fit(x_train,y_train)
print('poly : ' , model.score(x_test,y_test)) 

scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring= 'accuracy')
print('polyCV : ', scores)
print('polyCV 엔빵 : ', np.mean(scores))




