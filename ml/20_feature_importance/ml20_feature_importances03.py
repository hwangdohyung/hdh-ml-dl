import numpy as np 
from sklearn.datasets import load_diabetes
import pandas as pd

#1.데이터 
datasets = load_diabetes()
x = datasets.data 
y = datasets.target 

print(x.shape,y.shape)#(442, 10) (442,)

x= np.delete(x,1,axis=1)

print(x.shape,y.shape)#(442, 9) (441,)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8,random_state=123,shuffle=True)

#2.모델구성 
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor 
from xgboost import XGBClassifier,XGBRegressor

# model = DecisionTreeRegressor()
model = RandomForestRegressor()
# model = GradientBoostingRegressor()
# model = XGBRegressor()

#3.훈련 
model.fit(x_train,y_train)

#4.평가, 예측 
result = model.score(x_test,y_test)
print("model.score : ", result)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2_score : ', r2)

print('================================================')
print(model,':',model.feature_importances_) # 열의 중요도를 나타냄. 필요없는 컬럼 뻬기위함. (무조건 뺀다고 좋은것이 아님)

