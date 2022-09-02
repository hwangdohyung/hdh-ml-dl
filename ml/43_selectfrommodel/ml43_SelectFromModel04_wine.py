import numpy as np 
from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
import time 
from sklearn.metrics import accuracy_score,r2_score
from sklearn.feature_selection import SelectFromModel

#1.데이터 
datasets = load_wine()
x = datasets.data
y = datasets.target 
print(x.shape, y.shape) #(178, 13) (178,)


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=123, 
                                                 stratify=y
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

#2.모델 
model = XGBClassifier(random_state = 123,
                      n_estimators = 100,
                      learning_rate = 0.1,
                      max_depth = 3,
                      gamma = 1, 
                      )

# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

model.fit(x_train,y_train,early_stopping_rounds=20,
            eval_set =[(x_train,y_train),(x_test,y_test)]
             
        )  

results = model.score(x_test,y_test)
print(results)

print(model.feature_importances_)

thresholds = model.feature_importances_
print('================================')
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)  #prefit=true: feature importance 자기보다 같거나 큰놈들을 전부 반환해준다
    
    select_x_train = selection.transform(x_train)
    select_x_test= selection.transform(x_test)
    print(select_x_train.shape,select_x_test.shape)

    selection_model = XGBClassifier(random_state = 123,
                                    n_estimators = 100,
                                    learning_rate = 0.1,
                                    max_depth = 3,
                                    gamma = 1, 
                                    )
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    
    print("Thres=%.3f, n=%d, acc: %.2f%%"
          %(thresh, select_x_train.shape[1], score*100))

# 0.9166666666666666

# (142, 5) (36, 5)
# Thres=0.107, n=5, acc: 94.44%