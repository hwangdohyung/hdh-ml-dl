


from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_wine,fetch_covtype
from sklearn.model_selection import cross_val_predict, train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score

#1.데이터
datasets = fetch_covtype()
x = datasets['data']
y = datasets['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=72)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle= True, random_state=66)  # 5개 중에 1 개를 val 로 쓰겠다. 교차검증!

#2.모델구성
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3.4.컴파일,훈련,평가,예측
# model.fit(x_train,y_train)
scores = cross_val_score(model, x_train, y_train, cv= kfold)
print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test,y_predict)
print('cross_val_predict ACC : ', acc)

# ACC :  [0.92 0.96 1.   0.96 1.  ] 
#  cross_val_score :  0.968
# cross_val_predict ACC :  0.9444444444444444