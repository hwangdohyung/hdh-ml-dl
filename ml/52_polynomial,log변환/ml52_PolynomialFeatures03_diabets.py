from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline 

#1.데이터 
datasets = load_diabetes()
x, y = datasets.data, datasets.target 
print(x.shape,y.shape) 

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True,random_state=1234,)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

#2.모델 
model = make_pipeline(StandardScaler(),LinearRegression(),)
model.fit(x_train,y_train)
print('그냥: ' , model.score(x_test,y_test)) 

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring= 'r2')
print('그냥CV : ', scores)
print('그냥CV 엔빵 : ', np.mean(scores))


##################### polynomial 후 ########################

pf = PolynomialFeatures(degree=2, include_bias=False)

xp = pf.fit_transform(x)
print(xp.shape)             #(506, 105) 

x_train,x_test,y_train,y_test = train_test_split(xp,y, train_size=0.8, shuffle=True,random_state=1234,)

#2.모델 
model = make_pipeline(StandardScaler(),LinearRegression(),)
model.fit(x_train,y_train)
print('poly : ' , model.score(x_test,y_test)) 

scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring= 'r2')
print('polyCV : ', scores)
print('polyCV 엔빵 : ', np.mean(scores))

# (442, 10) (442,)
# 그냥:  0.46263830098374936
# 그냥CV :  [0.53864643 0.43212505 0.51425541 0.53344142 0.33325728]
# 그냥CV 엔빵 :  0.47034511859358785
# (442, 65)
# poly :  0.4186731903894866
# polyCV :  [0.48845419 0.27385747 0.28899779 0.26214683 0.10885984]
# polyCV 엔빵 :  0.2844632257068639