import numpy as np 
import pandas as pd 
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Input,LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
from collections import Counter
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import datetime as dt
from sklearn.model_selection import KFold,GridSearchCV
from xgboost import XGBRegressor,XGBClassifier


################## 1.데이터 #######################
path = 'D:\study_data\_data\dacon_travle/'
train_set = pd.read_csv(path + 'train.csv', index_col=0) 
print(train_set)
print(train_set.shape)  #(1955, 19)
test_set = pd.read_csv(path + 'test.csv', index_col=0)  
print(test_set.shape)   #(2933, 18)


#################### 결측치 #####################
train_set = train_set.dropna()
# test_set = test_set.dropna()
print(train_set.shape,test_set.shape)#(1649, 19) (2479, 18)
##################### 맵핑 ######################
test_set['Designation'] = test_set['Designation'].map({'Executive':0,'Manager':1,'Senior Manager':2,'AVP':3,'VP':4})
test_set['MaritalStatus'] = test_set['MaritalStatus'].map({'Married':0,'Divorced':1,'Unmarried':2,'Single':3})
test_set['ProductPitched'] = test_set['ProductPitched'].map({'Basic':0,'Deluxe':1,'Standard':2,'Super Deluxe':3,'King':4})
test_set['Gender'] = test_set['Gender'].map({'Male':0,'Fe Male':1,'Female':1})
test_set['Occupation'] = test_set['Occupation'].map({'Small Business':0,'Salaried':1,'Large Business':2,'Free Lancer':3})
test_set['TypeofContact'] = test_set['TypeofContact'].map({'Company Invited':0,'Self Enquiry':1})

train_set['Designation'] = train_set['Designation'].map({'Executive':0,'Manager':1,'Senior Manager':2,'AVP':3,'VP':4})
train_set['MaritalStatus'] = train_set['MaritalStatus'].map({'Married':0,'Divorced':1,'Unmarried':2,'Single':3})
train_set['ProductPitched'] = train_set['ProductPitched'].map({'Basic':0,'Deluxe':1,'Standard':2,'Super Deluxe':3,'King':4})
train_set['Gender'] = train_set['Gender'].map({'Male':0,'Fe Male':1,'Female':1})
train_set['Occupation'] = train_set['Occupation'].map({'Small Business':0,'Salaried':1,'Large Business':2,'Free Lancer':3})
train_set['TypeofContact'] = train_set['TypeofContact'].map({'Company Invited':0,'Self Enquiry':1})





# from datetime import datetime, timedelta
# from pandas import DataFrame
# import matplotlib.pyplot as plt 
# import seaborn as sns
# sns.set(font_scale= 1.2)
# sns.heatmap(data=train_set.corr(), square= True, annot=True, cbar=True) 

# plt.show()

x = train_set.drop('ProdTaken',axis=1)
y = train_set['ProdTaken']
print(x.shape)
print(x)
print(y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=123,stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state = 123)

parameters = {'n_estimators' : [100],'learning_rate': [0.2],'max_depth': [2],'gamma': [100],'min_child_weight': [10], 'subsample': [0.1],
              'colsample_bytree': [0],'colsample_bylevel': [0],'colsample_bynode': [0] ,'reg_alpha': [2],'reg_lambda':[10]
              }


#2.모델 
xgb = XGBClassifier(random_state = 123)

model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

model.fit(x_train,y_train)

print('최상의 매개변수 : ', model.best_params_)
print('최상의 점수 : ', model.best_score_)

results = model.score(x_test,y_test)
print(results)

y_submmit = model.predict(test_set)

submission = pd.read_csv(path + 'sample_submission.csv')
submission['ProdTaken'] = y_submmit

# print(submission)
# print(submission.shape)

submission.to_csv(path + 'submission.csv',index=False)