from sklearn.datasets import load_boston
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
datasets = load_boston()
x, y = datasets.data, datasets.target 
print(x.shape,y.shape) 

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True,random_state=1234,)

# scaler = StandardScaler() 
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = MinMaxScaler()    
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = MaxAbsScaler()    
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = RobustScaler()     
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = QuantileTransformer()  #
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = PowerTransformer(method='yeo-johnson')  # 디폴트   
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

scaler = PowerTransformer(method='box-cox')
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델 
# model = LinearRegression()
model = RandomForestRegressor()

#3.훈련
model.fit(x_train,y_train)

#4.평가,예측
y_predict = model.predict(x_test)
results = r2_score(y_test,y_predict)
print('기냥 결과 : ', round(results, 4))
#lr기냥 결과 :  0.7665  rf기냥 결과 :  0.9181

##############로그 변환 ##################
# df = pd.DataFrame(datasets.data,columns=[datasets.feature_names])
# print(df)

# # df.plot.box()
# # plt.title('boston')
# # plt.xlabel('Feature')
# # plt.ylabel('데이터값')
# # plt.show()

# print(df['B'].head())
# #         B
# # 0  396.90
# # 1  396.90
# # 2  392.83
# # 3  394.63
# # 4  396.90

# df['B'] = np.log1p(df['B'])      
# # print(df['B'].head())
# #           B
# # 0  5.986201
# # 1  5.986201
# # 2  5.975919
# # 3  5.980479
# # 4  5.986201

# # df['CRIM'] = np.log1p(df['CRIM']) #log변환 결과 :  0.7596
# df['ZN'] = np.log1p(df['ZN'])     #log변환 결과 :  0.7734
# df['TAX'] = np.log1p(df['TAX'])     #log변환 결과 :  0.7669

# x_train,x_test,y_train,y_test = train_test_split(df,y, train_size=0.8, shuffle=True,random_state=1234,)

# # scaler = StandardScaler()
# # x_train = scaler.fit_transform(x_train)
# # x_test = scaler.transform(x_test)

# #2.모델 
# model = LinearRegression()
# # model = RandomForestRegressor()

# #3.훈련
# model.fit(x_train,y_train)

# #4.평가,예측
# y_predict = model.predict(x_test)
# results = r2_score(y_test,y_predict)
# print('log변환 결과 : ', round(results, 4))

# #lr log변환 결과 :  0.7711  ,0.9213

# # log변환 결과 :  0.7785


