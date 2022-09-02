from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split,KFold,cross_val_score,cross_val_predict
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler


#1.데이터
datasets = load_diabetes()

x = datasets['data']
y = datasets['target']
print(x.shape, y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)


scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 

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

# R2 :  [0.35212556 0.49142493 0.26340135 0.31775697 0.2142026 ] 
# cross_val_score :  0.3278
# cross_val_predict r2 :  0.49427991175696084