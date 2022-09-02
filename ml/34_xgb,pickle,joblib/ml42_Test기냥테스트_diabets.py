import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
import time 
from sklearn.metrics import accuracy_score,r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression

#1.데이터 
datasets = load_diabetes()
x = datasets.data
y = datasets.target 
print(x.shape, y.shape) #(442, 10) (442,)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=72, 
                                                #  stratify=y
                                                )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 123)

parameters = {'n_estimators' : [100],'learning_rate': [0.1],'max_depth': [3],'gamma': [1],
              'min_child_weight': [1],'subsample': [1],'colsample_bytree': [1],'colsample_bylevel': [1],
              'colsample_bynode': [1] ,'reg_alpha': [0],'reg_lambda':[1]
              }

# #2.모델 
# model = XGBRegressor(random_state = 123,
#                       n_estimators = 100,
#                       learning_rate = 0.1,
#                       max_depth = 3,
#                       gamma = 1, 
#                       )

# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

model = LinearRegression()
# model.fit(x_train,y_train,early_stopping_rounds=200,
#             eval_set =[(x_train,y_train),(x_test,y_test)],
#             eval_metric = 'error' 
#           )  
model.fit(x_train,y_train)

results = model.score(x_test,y_test)
print(results)
'''
print(model.feature_importances_)
# [0.03986917 0.04455113 0.25548902 0.07593288 0.04910125 0.04870857
#  0.06075545 0.05339111 0.30488744 0.06731401]

thresholds = model.feature_importances_
print('================================')
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)  #prefit=true: feature importance 자기보다 같거나 큰놈들을 전부 반환해준다
    
    select_x_train = selection.transform(x_train)
    select_x_test= selection.transform(x_test)
    print(select_x_train.shape,select_x_test.shape)

    selection_model = XGBRegressor(random_state = 123,
                                    n_estimators = 100,
                                    learning_rate = 0.1,
                                    max_depth = 3,
                                    gamma = 1, 
                                    )
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    
    print("Thres=%.3f, n=%d, R2: %.2f%%"
          %(thresh, select_x_train.shape[1], score*100))
'''
# 0.6579209558684549

