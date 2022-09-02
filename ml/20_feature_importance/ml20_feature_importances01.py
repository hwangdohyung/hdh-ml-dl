import numpy as np 
from sklearn.datasets import load_iris 


#1.데이터 
datasets = load_iris()
x = datasets.data 
y = datasets.target 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8,random_state=123,shuffle=True)

#2.모델구성 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from xgboost import XGBClassifier

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()

#3.훈련 
model.fit(x_train,y_train)

#4.평가, 예측 
result = model.score(x_test,y_test)
print("model.score : ", result)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print('accuracy_score : ', acc)

print('================================================')
print(model,':',model.feature_importances_) # 열의 중요도를 나타냄. 필요없는 컬럼 뻬기위함. 

# DecisionTreeClassifier() : [0.01088866 0.04177982 0.0634409  0.88389062]
# RandomForestClassifier() : [0.10372951 0.02944583 0.43822629 0.42859837]
# GradientBoostingClassifier() : [0.00080998 0.02430679 0.64768496 0.32719826]
# XGBClassifier():[0.0089478  0.01652037 0.75273126 0.22180054]







