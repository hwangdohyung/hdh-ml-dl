


from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_predict, train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score

#1.데이터
datasets = load_boston()
x = datasets['data']
y = datasets['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=72)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle= True, random_state=66)  # 5개 중에 1 개를 val 로 쓰겠다. 교차검증!

#2.모델구성
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

#3.4.컴파일,훈련,평가,예측
# model.fit(x_train,y_train)
scores = cross_val_score(model, x_train, y_train, cv= kfold)
print('R2 : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
r2 = r2_score(y_test,y_predict)
print('cross_val_predict r2 : ', r2)

# R2 :  [0.82981587 0.93816037 0.84398156 0.78081615 0.93181948] 
#  cross_val_score :  0.8649
# cross_val_predict r2 :  0.7031717009263274