from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,MinMaxScaler,MaxAbsScaler,RobustScaler,QuantileTransformer,PowerTransformer
import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline 
from sklearn.metrics import r2_score,accuracy_score
from xgboost import XGBClassifier,XGBRegressor
import matplotlib.pyplot as plt 

#1.데이터 
datasets = load_diabetes()
x, y = datasets.data, datasets.target 
print(x.shape,y.shape) 

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True,random_state=1234)

scaler_li = [StandardScaler(),MinMaxScaler(),MaxAbsScaler(),RobustScaler(),QuantileTransformer(),PowerTransformer(method='yeo-johnson')]

for i in scaler_li:
    scaler = i
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = XGBRegressor(random_state=123,n_estimators =100,
              learning_rate= 0.2,
              max_depth= 2 ,
              gamma= 100,
              min_child_weight=10,
              subsample=0.1,
              colsample_bytree=0,
              colsample_bylevel=0,
              colsample_bynode=0,
              reg_alpha=2,
              reg_lambda=10)
    model.fit(x_train,y_train)

    y_predict = model.predict(x_test)
    print('결과 : ',round(r2_score(y_test,y_predict),4))


# scaler = PowerTransformer(method='yeo-johnson')  # 디폴트    
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = PowerTransformer(method='box-cox')
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)




