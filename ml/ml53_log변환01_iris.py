from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,MinMaxScaler,MaxAbsScaler,RobustScaler,QuantileTransformer,PowerTransformer
import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline 
from sklearn.metrics import r2_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import matplotlib.pyplot as plt 

#1.데이터 
datasets = load_iris()
x, y = datasets.data, datasets.target 
print(x.shape,y.shape) 

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True,random_state=1234,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델 
model = RandomForestClassifier()

#3.훈련
model.fit(x_train,y_train)

#4.평가,예측
y_predict = model.predict(x_test)
results = r2_score(y_test,y_predict)
print('기냥 결과 : ', round(results, 4))
#lr기냥 결과 :  0.7665  rf기냥 결과 :  0.9181

##############로그 변환 ##################
df = pd.DataFrame(datasets.data,columns=[datasets.feature_names])
print(df)



# print(df['B'].head())

df = np.log1p(df)      

# print(df['B'].head())


x_train,x_test,y_train,y_test = train_test_split(df,y, train_size=0.8, shuffle=True,random_state=1234,)


#2.모델 
model = RandomForestClassifier()

#3.훈련
model.fit(x_train,y_train)

#4.평가,예측
y_predict = model.predict(x_test)
results = r2_score(y_test,y_predict)
print('log변환 결과 : ', round(results, 4))

