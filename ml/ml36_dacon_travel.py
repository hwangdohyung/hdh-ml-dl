import numpy as np 
import pandas as pd 
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Input,LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
from collections import Counter
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import datetime
from sklearn.model_selection import KFold,GridSearchCV
from xgboost import XGBRegressor
################## 1.데이터 #######################
path = 'D:\study_data\_data\dacon_travle/'
train_set = pd.read_csv(path + 'train.csv', index_col=0) 
# print(train_set)
# print(train_set.shape) #(6255, 12)
test_set = pd.read_csv(path + 'test.csv', index_col=0)  
# print(test_set.shape)

#################### 결측치 ########################
# print(train_set.isnull().sum())
# print(test_set.isnull().sum())
means_train = train_set.mean() 
train_set = train_set.fillna(means_train)
means_test = train_set.mean() 
test_set = test_set.fillna(means_test)
################## 이상치 ######################
# def outlier(train_set):
#     quartile_1 = train_set.quantile(0.25)
#     quartile_3 = train_set.quantile(0.75)
#     IQR = quartile_3 - quartile_1
#     condition = (train_set < (quartile_1 - 1.5 * IQR)) | (train_set > (quartile_3 + 1.5 * IQR))
#     condition = condition.any(axis=1)

#     return train_set, train_set.drop(train_set.index, axis=0)

# outlier(train_set)

# import matplotlib.pyplot as plt 
# import seaborn as sns
# sns.set(font_scale= 1.2)
# sns.heatmap(data=train_set.corr(), square= True, annot=True, cbar=True) 

# plt.show()

x = train_set.drop('Weekly_Sales',axis=1)
y = train_set['Weekly_Sales']
print(x.shape)
print(x)
print(y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=123)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state = 123)

parameters = {'n_estimators' : [100],'learning_rate': [0.2],'max_depth': [2],'gamma': [100],'min_child_weight': [10], 'subsample': [0.1],
              'colsample_bytree': [0],'colsample_bylevel': [0],'colsample_bynode': [0] ,'reg_alpha': [2],'reg_lambda':[10]
              }

#2.모델 
xgb = XGBRegressor(random_state = 123)

model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

model.fit(x_train,y_train)

print('최상의 매개변수 : ', model.best_params_)
print('최상의 점수 : ', model.best_score_)

results = model.score(x_test,y_test)
print(results)

