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
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import KFold,GridSearchCV,StratifiedKFold,HalvingRandomSearchCV
from xgboost import XGBRegressor,XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline 


################## 1.데이터 #######################
path = 'D:\study_data\_data\dacon_travle/'
train_set = pd.read_csv(path + 'train.csv', index_col=0) 
print(train_set)
print(train_set.shape)  #(1955, 19)
test_set = pd.read_csv(path + 'test.csv', index_col=0)  
print(test_set.shape)   #(2933, 18)

#################### 결측치 #####################
train_set = train_set.dropna()

means = test_set.mean() #컬럼별 평균
test_set = test_set.fillna(means)



print(train_set.shape,test_set.shape)#(1649, 19) (2479, 18)
##################### 라벨인코더 ######################
le = LabelEncoder()

idxarr = train_set.columns
idxarr = np.array(idxarr)

for i in idxarr:
      if train_set[i].dtype == 'object':
        train_set[i] = le.fit_transform(train_set[i])
        test_set[i] = le.fit_transform(test_set[i])
        
print(train_set)
print(test_set) 
       
       
# from datetime import datetime, timedelta
# from pandas import DataFrame
# import matplotlib.pyplot as plt 
# import seaborn as sns
# sns.set(font_scale= 1.0)
# sns.heatmap(data=train_set.corr(), square= True, annot=True, cbar=True) 

# plt.show()

x = train_set.drop('ProdTaken',axis=1)
y = train_set['ProdTaken']


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=134,stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# parameters = {'n_estimators' : [100],'learning_rate': [0.2],'max_depth': [2],'gamma': [100],'min_child_weight': [10], 'subsample': [0.1],
#               'colsample_bytree': [0],'colsample_bylevel': [0],'colsample_bynode': [0] ,'reg_alpha': [2],'reg_lambda':[10]
#               }

parameters = [
    {'n_estimators':[100,200]},
    {'max_depth':[6,8,10,12]},
    {'min_samples_leaf':[3,5,7,10]},
    {'min_samples_split':[2,3,5,10]},
    {'n_jobs':[-1,2,4]}]

#2.모델 
random= RandomForestClassifier()

model = make_pipeline(MinMaxScaler(), HalvingRandomSearchCV(random, parameters, cv=5, n_jobs=-1, verbose=2))

model.fit(x_train,y_train)


results = model.score(x_test,y_test)
print(results)

y_submmit = model.predict(test_set)
print(y_submmit)

print(pd.value_counts(y_submmit))

submission = pd.read_csv(path + 'sample_submission.csv')
submission['ProdTaken'] = y_submmit

# print(submission)
# print(submission.shape)

submission.to_csv(path + 'submission.csv',index=False)



