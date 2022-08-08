import numpy as np 
from sklearn.datasets import load_diabetes


#1.데이터 
datasets = load_diabetes()
x = datasets.data 
y = datasets.target 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8,random_state=123,shuffle=True)

#2.모델구성 
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor 
from xgboost import XGBClassifier,XGBRegressor

# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
model = GradientBoostingRegressor()
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
print(model,':',model.feature_importances_) # 열의 중요도를 나타냄. 필요없는 컬럼 뻬기위함. 

# DecisionTreeRegressor() : [0.0963733  0.01927495 0.23200568 0.05365728 0.04820936  0.06357137 0.0384896  0.0113324  0.36526691 0.07181913]
# model.score :  0.14524475636299472
# r2_score :  0.14524475636299472

# RandomForestRegressor() : [0.05828263 0.01187727 0.28179166 0.10187361 0.04208812 0.05648166 0.05263341 0.0289396  0.28467408 0.08135795]
# model.score :  0.5267853266754605
# r2_score :  0.5267853266754605

# GradientBoostingRegressor() : [0.04976515 0.01081618 0.30407614 0.11147504 0.0272528  0.05514144 0.04069048 0.01872486 0.33810916 0.04394875]
# model.score :  0.553847636169599
# r2_score :  0.553847636169599

# XGBRegressor() : [0.03234756 0.0447546  0.21775807 0.08212128 0.04737141 0.04843819 0.06012432 0.09595273 0.30483875 0.06629313]
# model.score :  0.4590400803596264
# r2_score :  0.4590400803596264

