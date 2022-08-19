#[실습]#
from re import I
import numpy as np 
import pandas as pd
from sklearn import metrics
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Dropout,Input,Flatten,Conv2D
from sklearn.experimental import enable_halving_search_cv # 실험적 버전 정식버전이 아니다 *
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split,GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import tensorflow as tf
import datetime
from sklearn.model_selection import cross_val_score,cross_val_predict,KFold,StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

#1.데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv',index_col =0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

##########전처리############
train_test_data = [train_set, test_set]
sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

print(dataset)

for dataset in train_test_data:
    # 가족수 = 형제자매 + 부모님 + 자녀 + 본인
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    
    # 가족수 > 1이면 동승자 있음
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
embarked_mapping = {'S':0, 'C':1, 'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([\w]+)\.', expand=False)
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].apply(lambda x: 0 if x=="Mr" else 1 if x=="Miss" else 2 if x=="Mrs" else 3 if x=="Master" else 4)

train_set['Cabin'] = train_set['Cabin'].str[:1]
for dataset in train_test_data:
    dataset['Age'].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset['Agebin'] = pd.cut(dataset['Age'], 5, labels=[0,1,2,3,4])
for dataset in train_test_data:
    dataset["Fare"].fillna(dataset.groupby("Pclass")["Fare"].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset['Farebin'] = pd.qcut(dataset['Fare'], 4, labels=[0,1,2,3])
    drop_column = ['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']

for dataset in train_test_data:
    dataset = dataset.drop(drop_column, axis=1, inplace=True)
print(train_set.head())

from xgboost import XGBClassifier
from sklearn.preprocessing import QuantileTransformer,PowerTransformer
x = train_set.drop(['Survived'], axis=1,)
y = train_set['Survived']


from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor,LGBMClassifier

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True,random_state=1234,stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


bayesian_params ={'max_depth':(2,16),'gamma': (0,100),'min_child_weight':(1,50),'subsample':(0.1,1),
                  'colsample_bytree':(0.1,1),'colsample_bylevel':(0.1,1),'colsample_bynode':(0.1,1),'max_bin':(10,500),'reg_lambda':(0.001,10),'reg_alpha':(0.01,50)}


def lgb_hamsu(max_depth, gamma, min_child_weight,subsample,colsample_bytree,colsample_bylevel,colsample_bynode,max_bin, reg_lambda, reg_alpha):
    params ={'n_estimators':300, 'learning_rate':0.1,
             'max_depth':int(round(max_depth)),                  # 무조건 정수
             'gamma': int(round(gamma)),
             
             'min_child_weight': int(round(min_child_weight)),  
             'subsample': max(min(subsample,1),0),              # 어떤 값을 넣어도 0~1 의 값
             'colsample_bytree': max(min(colsample_bytree,1),0),
             'colsample_bylevel': max(min(colsample_bylevel,1),0),
             'colsample_bynode': max(min(colsample_bynode,1),0),
             'max_bin': max(int(round(max_bin)),10),            # 무조건 10이상
             'reg_lambda': max(reg_lambda,0),                   # 무조건 0이상(양수)
             'reg_alpha':max(reg_alpha,0)                       
    }

    #  * : 여러개의인자를받겠다
    # ** : 키워드받겠다(딕셔너리형태)
    model = XGBClassifier(**params) 

    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
            #   eval_metric='merror',
              verbose=0,
              early_stopping_rounds=50)

    y_predict = model.predict(x_test)
    results = accuracy_score(y_test,y_predict)

    return results

lgb_bo = BayesianOptimization(f=lgb_hamsu,
                              pbounds=bayesian_params,
                              random_state=123)

lgb_bo.maximize(init_points=5, n_iter=100)  #초기 2번  ,n-iter : 20번 돌거다! 총 22번 돈다 

print(lgb_bo.max) 

# {'target': 0.9833333333333333, 'params': {'colsample_bylevel': 1.0, 'colsample_bynode': 1.0, 'colsample_bytree': 1.0,
#                                           'gamma': 0.0, 'max_bin': 179.7102064586691, 'max_depth': 2.0, 'min_child_weight': 1.0, 
#                                           'reg_alpha': 0.01, 'reg_lambda': 0.001, 'subsample': 1.0}}