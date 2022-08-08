import numpy as np 
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler,StandardScaler

#1.데이터 
datasets = load_boston() 
x = datasets.data 
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size= 0.8, shuffle=True, random_state=1234)                                                    

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델 
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from sklearn.pipeline import make_pipeline

# model = RandomForestClassifier()
model = make_pipeline(MinMaxScaler(), RandomForestRegressor())

#3.훈련 
model.fit(x_train, y_train) # 파이프라인에서 fit 할땐 스케일링의 transform 과 fit이 돌아간다. 

#4.평가, 예측 
result = model.score(x_test, y_test)

print('model.r2 : ', round(result,2))

# model.r2 :  0.91






