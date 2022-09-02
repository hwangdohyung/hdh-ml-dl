# 데이콘 따릉이 문제풀이 
import numpy as np 
import pandas as pd # csv 파일 당겨올 때 사용
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error 
from collections import Counter
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LinearRegression
 
 
#1.데이터 
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col=0) #컬럼중에 id컬럼(0번째)은 단순 index 

test_set = pd.read_csv(path + 'test.csv', index_col=0)  #예측에서 쓴다!


#####결측치 처리 1. 제거 ######
# print(train_set.isnull().sum()) 
train_set = train_set.dropna()
# print(train_set.isnull().sum()) 
# print(train_set.shape)
test_set= test_set.fillna(test_set.mean())
##############################

x = train_set.drop(['count'], axis=1,)

y = train_set['count']

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
print(xp.shape)             

x_train,x_test,y_train,y_test = train_test_split(xp,y, train_size=0.8, shuffle=True,random_state=123)

#2.모델 
model = make_pipeline(StandardScaler(),LinearRegression(),)
model.fit(x_train,y_train)
print('poly : ' , model.score(x_test,y_test))

scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring= 'r2')
print('polyCV : ', scores)
print('polyCV 엔빵 : ', np.mean(scores))

# 그냥:  0.5847384872144868
# 그냥CV :  [0.55090118 0.62069405 0.5838313  0.59093011 0.56302352]
# 그냥CV 엔빵 :  0.5818760327045374
# (1328, 54)
# poly :  0.6480625224203171
# polyCV :  [0.58493206 0.64697857 0.55873654 0.62055164 0.66976466]
# polyCV 엔빵 :  0.6161926917063152