from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor,LGBMClassifier
import numpy as np 
from xgboost import XGBClassifier,XGBRegressor

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
import warnings
warnings.filterwarnings('ignore')

#1.데이터 
datasets = load_iris()
x,y = datasets.data, datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8,random_state=123,shuffle=True,stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


bayesian_params ={'max_depth':(6,16),'num_leaves': (24,64),'min_child_samples':(10,200),'min_child_weight':(1,50),'subsample':(0.5,1),
                  'colsample_bytree':(0.5,1),'max_bin':(10,500),'reg_lambda':(0.001,10),'reg_alpha':(0.01,50)}


def lgb_hamsu(max_depth, num_leaves, min_child_samples, min_child_weight,subsample,colsample_bytree,max_bin, reg_lambda, reg_alpha):
    params ={'n_estimators':500, 'learning_rate':0.02,
             'max_depth':int(round(max_depth)),                  # 무조건 정수
             'num_leaves': int(round(num_leaves)),
             'min_child_samples': int(round(min_child_samples)),
             'min_child_weight': int(round(min_child_weight)),  
             'subsample': max(min(subsample,1),0),              # 어떤 값을 넣어도 0~1 의 값
             'colsample_bytree': max(min(colsample_bytree,1),0),
             'max_bin': max(int(round(max_bin)),10),            # 무조건 10이상
             'reg_lambda': max(reg_lambda,0),                   # 무조건 0이상(양수)
             'reg_alpha':max(reg_alpha,0)                       
    }

    #  * : 여러개의인자를받겠다
    # ** : 키워드받겠다(딕셔너리형태)
    model = LGBMClassifier(**params) 

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
# {'target': 0.9666666666666667, 'params': {'colsample_bytree': 1.0, 'max_bin': 245.56803758419673, 'max_depth': 16.0, 
# 'min_child_samples': 10.0, 'min_child_weight': 1.0, 'num_leaves': 64.0, 'reg_alpha': 0.01, 'reg_lambda': 10.0, 'subsample': 1.0}}




