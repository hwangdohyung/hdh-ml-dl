from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,MinMaxScaler,MaxAbsScaler,RobustScaler,QuantileTransformer,PowerTransformer
import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline 
from sklearn.metrics import r2_score,accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt 

#1.데이터 
datasets = load_digits()
x, y = datasets.data, datasets.target 
print(x.shape,y.shape) 

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True,random_state=1234,stratify=y)

scaler_li = [StandardScaler(),MinMaxScaler(),MaxAbsScaler(),RobustScaler(),QuantileTransformer(),PowerTransformer(method='yeo-johnson')]

for i in scaler_li:
    scaler = i
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = XGBClassifier(random_state=123,n_estimators =200,
              learning_rate= 0.1,
              max_depth= 3,
              gamma= 0,
              min_child_weight=1,
              subsample=0.7,
              colsample_bytree=0.3,
              colsample_bylevel=0.5, 
              colsample_bynode=1 ,
              reg_alpha=0.001,
              reg_lambda=0.001)
    model.fit(x_train,y_train)

    y_predict = model.predict(x_test)
    print('결과 : ',round(accuracy_score(y_test,y_predict),4))


# scaler = PowerTransformer(method='yeo-johnson')  # 디폴트    
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = PowerTransformer(method='box-cox')
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


