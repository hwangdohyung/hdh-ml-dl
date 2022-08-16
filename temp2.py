
import pandas as pd
import random
import os
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier,XGBRegressor

path = 'D:\study_data\_data\dacon_antena/'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) 



train = pd.read_csv(path + 'train.csv',index_col=0)
test = pd.read_csv(path + 'test.csv',index_col=0)
submit = pd.read_csv(path + 'sample_submission.csv',index_col=0)

print(train.shape,test.shape)

alldata = pd.concat((train, test), axis=0)
alldata_index = alldata.index

alldata = alldata.drop(['X_04','X_10','X_11','X_23','X_47','X_48'],axis=1)

print(alldata.shape)
train = alldata[:len(train)]
test = alldata[len(train):]

print(test)
print(train.shape,test.shape,submit.shape)


x = train.filter(regex='X') # Input : X Featrue
y = train.filter(regex='Y') # Output : Y Feature
test = test.filter(regex='X') # Output : Y Feature

print(x.shape,y.shape,test.shape,submit.shape)
print(x)
print(test)


from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, 
#                                                      random_state=123)

from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler
scaler = MaxAbsScaler()
x = scaler.fit_transform(x)
test = scaler.transform(test)

#2. 모델 
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline 
from xgboost import XGBRegressor

model = XGBRegressor(random_state = 66)

model.fit(x,y)

result = model.score(x,y)
print('결과 : ', result)


submit= model.predict(test)

submission = pd.read_csv(path + 'sample_submission.csv')

submission[['Y_01','Y_02','Y_03','Y_04','Y_05','Y_06','Y_07','Y_08','Y_09','Y_10','Y_11','Y_12','Y_13','Y_14']] = submit
print('=============================')
print(submission)
print(submission.shape)

submission= pd.DataFrame(submission)
print(submission)
print(submission.shape)

submission.to_csv(path +'last1.csv',index=False)

