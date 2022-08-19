from sklearn.datasets import fetch_covtype
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
datasets = fetch_covtype()
x, y = datasets.data, datasets.target 
print(x.shape,y.shape) 

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True,random_state=1234,stratify=y)

scaler_li = [StandardScaler(),MinMaxScaler(),MaxAbsScaler(),RobustScaler(),QuantileTransformer(),PowerTransformer(method='yeo-johnson')]

for i in scaler_li:
    scaler = i
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = XGBClassifier()
    model.fit(x_train,y_train)

    y_predict = model.predict(x_test)
    print('결과 : ',round(accuracy_score(y_test,y_predict),4))


# scaler = PowerTransformer(method='yeo-johnson')  # 디폴트    
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = PowerTransformer(method='box-cox')
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


