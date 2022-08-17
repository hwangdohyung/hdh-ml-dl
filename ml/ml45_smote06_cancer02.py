# 1 357
# 0 212
# 라벨 0을 112개 삭제해서 재구성
# smote 해서 비교 
# 넣은거 안넣은거 비교 
# acc,f1 으로 비교 2진분류니까 macro 필요x

#실습 시작!! 
from dataclasses import replace
import pandas as pd 
import numpy as np 
from sklearn.datasets import load_breast_cancer
# 1.데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target


zero = np.where(y==0)
zero= zero[0][:112] 
x = np.delete(x,zero,0)
y = np.delete(y,zero,0)

print(np.unique(y,return_counts=True))   #(array([0, 1]), array([100, 357], dtype=int64))
########


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8,random_state=123,shuffle=True,stratify=y)

# 2.모델구성
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# 3.훈련 
model.fit(x_train,y_train)

# 4.평가, 예측 
from sklearn.metrics import accuracy_score,f1_score
y_predict = model.predict(x_test)
score = model.score(x_test,y_test)

print('acc_score : ', accuracy_score(y_test,y_predict)) 
print('f1_score : ', f1_score(y_test,y_predict))


from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
print('=========== SMOTE 적용후 ==============')
smote = SMOTE(random_state=123,k_neighbors=4) 
x_train,y_train = smote.fit_resample(x_train,y_train)


model= RandomForestClassifier()
model.fit(x_train, y_train)     

y_predict = model.predict(x_test)
score = model.score(x_test,y_test)
# print('model.score: ', score)                           
print('acc_score : ', accuracy_score(y_test,y_predict)) 
print('f1_score : ', f1_score(y_test,y_predict))


# acc_score :  0.956140350877193
# f1_score :  0.9645390070921985
# =========== SMOTE 적용후 ==============
# acc_score :  0.9736842105263158
# f1_score :  0.9787234042553191

# 데이터 줄이고 다시 증폭 후
# acc_score :  0.967391304347826
# f1_score :  0.9793103448275863
# =========== SMOTE 적용후 ==============
# acc_score :  0.9782608695652174
# f1_score :  0.9861111111111112




