from operator import index
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split,KFold,RandomizedSearchCV,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier,XGBRegressor

datasets = pd.read_csv('D:/study_data/_data/winequality-white.csv',index_col=None,header=0,sep=';')

datasets = datasets.drop(['density','chlorides'],axis=1)

x = datasets.drop(('quality'),axis=1)
y = datasets['quality']

# import matplotlib.pyplot as plt 
# import seaborn as sns
# ### 상관관계 ###
# sns.set(font_scale= 0.8 )
# sns.heatmap(data=datasets.corr(), square= True, annot=True, cbar=True) # square: 정사각형, annot: 안에 수치들 ,cbar: 옆에 bar

# plt.show() 

le = LabelEncoder()

y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, 
                                                     random_state=72)

       
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state = 123)


parameters = {'n_estimators' : [300],
              'learning_rate': [0.3],
              'max_depth': [7],
              'gamma': [0],
              'min_child_weight': [1],
              'subsample': [1],
              'colsample_bytree': [1],
              'colsample_bylevel': [1],
              'colsample_bynode': [1] ,
              'reg_alpha': [0],
              'reg_lambda':[1]
              }

from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel


xgb = XGBClassifier(random_state = 123,
                    n_estimators = 300,
                    learning_rate = 0.3,
                    max_depth = 7,
                    gamma = 0,
                    min_child_weight = 1,
                    subsample = 1,
                    colsample_bytree=1,
                    colsample_bylevel=1,
                    colsample_bynode=1,
                    reg_alpha = 0,
                    reg_lambda = 1
                                   )
model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

model.fit(x_train,y_train)
# model.fit(x_train,y_train,early_stopping_rounds=10,
#             eval_set =[(x_train,y_train),(x_test,y_test)],
#         #   eval_set =[(x_test,y_test)]
#             eval_metric = 'merror'
#                 # 회귀: rmse, mae, rmsle ... 등등
#                 # 이진: error, logloss, auc... 등
#                 # 다중: merror, mlogloss... 등 
#           )  # 10번동안 갱신 없으면 정지 시키겠다


print('최상의 매개변수 : ', model.best_params_)
print('최상의 점수 : ', model.best_score_)

result = model.score(x_test,y_test)
print('결과 : ', result)

# 최상의 점수 :  0.6594866171970113
# 결과 :  0.7


