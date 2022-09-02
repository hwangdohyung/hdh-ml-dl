from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline 

#1.데이터 
datasets = fetch_california_housing()
x, y = datasets.data, datasets.target 
print(x.shape,y.shape) 
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True,random_state=123)

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

pf = PolynomialFeatures(degree=2, 
                         include_bias=False
                        )

xp = pf.fit_transform(x)
print(xp.shape)             #(506, 105) 

x_train,x_test,y_train,y_test = train_test_split(xp,y, train_size=0.8, shuffle=True,random_state=123)

#2.모델 
model = make_pipeline(StandardScaler(),LinearRegression(),)
model.fit(x_train,y_train)
print('poly : ' , model.score(x_test,y_test)) 

scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring= 'r2')
print('polyCV : ', scores)
print('polyCV 엔빵 : ', np.mean(scores))

# (20640, 8) (20640,)
# 그냥:  0.6104546894797875
# 그냥CV :  [0.62221973 0.61954582 0.56707402 0.60509591 0.02409723]
# 그냥CV 엔빵 :  0.4876065399222941
# (20640, 44)
# poly :  0.6028532846881773
# polyCV :  [  0.69901064   0.1989347    0.54220325   0.60128801 -23.28765387]
# polyCV 엔빵 :  -4.249243453910983





