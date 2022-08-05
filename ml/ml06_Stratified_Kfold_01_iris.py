#다중분류 모델에서 데이터가 적은경우 하나의 라벨이 너무 적게 나눠질수 있다. 라벨의 비율을 맞춰주는 stratified

from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold


#1.데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

# x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle= True, random_state=66)  # 5개 중에 1 개를 val 로 쓰겠다. 교차검증!
kfold = StratifiedKFold(n_splits=n_splits, shuffle= True, random_state=66)  # 5개 중에 1 개를 val 로 쓰겠다. 교차검증!

#2.모델구성
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3.4.컴파일,훈련,평가,예측
# model.fit(x_train,y_train)
scores = cross_val_score(model, x, y, cv= kfold)
print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))

# ACC :  [0.9        0.96666667 1.         0.9        0.96666667] 
# cross_val_score :  0.9467

