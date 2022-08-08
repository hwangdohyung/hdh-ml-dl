#실습 
# 피처인포턴스 전체 중요도에서 하위 20~25% 컬럼들을 제거하여 데이터셋 재구성후
# 각 모델별로 돌려서 결과 도출! 
# 기존 모델결과와 비교 

#2.모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

#결과비교 
#1. DecisionTree
#기존 acc: 
#컬럼 삭제후 acc :

import numpy as np 
from sklearn.datasets import load_digits


#1.데이터 
datasets = load_digits()
x = datasets.data 
y = datasets.target 
print(x.shape)

x = np.delete(x,(0,1,7,8,15,16,23,31,32,39,40),axis=1)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8,random_state=123,shuffle=True)

#2.모델구성 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from xgboost import XGBClassifier

model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()

#3.훈련 
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

#4.평가, 예측 
result1 = model1.score(x_test,y_test)
print("model1.score : ", result1)
result2 = model2.score(x_test,y_test)
print("model2.score : ", result2)
result3 = model3.score(x_test,y_test)
print("model3.score : ", result3)
result4 = model4.score(x_test,y_test)
print("model4.score : ", result4)



print(model1,':',np.round_(model1.feature_importances_,3)) # 열의 중요도를 나타냄. 필요없는 컬럼 뻬기위함.
print(model2,':',np.round_(model2.feature_importances_,3)) # 열의 중요도를 나타냄. 필요없는 컬럼 뻬기위함.
print(model3,':',np.round_(model3.feature_importances_,3)) # 열의 중요도를 나타냄. 필요없는 컬럼 뻬기위함.
print(model4,':',np.round_(model4.feature_importances_,3)) # 열의 중요도를 나타냄. 필요없는 컬럼 뻬기위함.

print(datasets.feature_names)

#삭제전
# model1.score :  0.8222222222222222
# model2.score :  0.9694444444444444
# model3.score :  0.9555555555555556
# model4.score :  0.9638888888888889

#삭제후
# model1.score :  0.8305555555555556
# model2.score :  0.9777777777777777
# model3.score :  0.9583333333333334
# model4.score :  0.9555555555555556

