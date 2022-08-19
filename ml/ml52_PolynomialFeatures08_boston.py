from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline 

#1.데이터 
datasets = load_boston()
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

# 그냥:  0.7665382927362877
# 그냥CV :  [0.71606004 0.67832011 0.65400513 0.56791147 0.7335664 ]
# 그냥CV 엔빵 :  0.669972627809433
# (506, 104)
# poly :  0.8745129304823845
# polyCV :  [0.7917776  0.8215846  0.79599441 0.81776798 0.81170102]
# polyCV 엔빵 :  0.8077651212215852