#실습 시작!! 
import pandas as pd 
import numpy as np 
from sklearn.datasets import load_breast_cancer
# 1.데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(np.unique(y,return_counts=True))    

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






