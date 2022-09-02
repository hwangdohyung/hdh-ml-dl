import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.experimental import enable_halving_search_cv # 실험적 버전 정식버전이 아니다 *
from sklearn.model_selection import train_test_split,KFold,GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV,StratifiedKFold
from sklearn.metrics import r2_score,mean_squared_error 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

#1.data 처리
############# id컬럼 index 처리 ##########
path = './_data/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')
train_set.set_index('Id', inplace=True)
test_set.set_index('Id', inplace=True)
test_id_index = train_set.index
trainLabel = train_set['SalePrice']
train_set.drop(['SalePrice'], axis=1, inplace=True)
############################################
################### 트레인,테스트 합치기 ##################
alldata = pd.concat((train_set, test_set), axis=0)
alldata_index = alldata.index
################## NA 값 20프로 이상은 drop! ##########
NA_Ratio = 0.8 * len(alldata)
alldata.dropna(axis=1, thresh=NA_Ratio, inplace=True)

################### 수치형,카테고리형 분리,범위 설정 #############
alldata_obj = alldata.select_dtypes(include='object') 
alldata_num = alldata.select_dtypes(exclude='object')

for objList in alldata_obj:
    alldata_obj[objList] = LabelEncoder().fit_transform(alldata_obj[objList].astype(str))
##################### 소수 na 값 처리 ###################    
imputer = SimpleImputer(strategy='mean')
imputer.fit(alldata_num)
alldata_impute = imputer.transform(alldata_num)
alldata_num = pd.DataFrame(alldata_impute, columns=alldata_num.columns, index=alldata_index)  
###################### 분리한 데이터 다시 합치기 #####################
alldata = pd.merge(alldata_obj, alldata_num, left_index=True, right_index=True)  
##################### 트레인, 테스트 다시 나누기 ##################
train_set = alldata[:len(train_set)]
test_set = alldata[len(train_set):]
############### 트레인 데이터에 sale price 합치기 ##############
train_set['SalePrice'] = trainLabel
############### sale price 다시 드랍 #####################
train_set = train_set.drop(['SalePrice'], axis =1)
print(train_set)
print(trainLabel)
print(test_set)

from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LinearRegression
 
x,y = train_set,trainLabel

from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer,PowerTransformer
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True,random_state=1234)

from bayes_opt import BayesianOptimization
from sklearn.preprocessing import QuantileTransformer,PowerTransformer

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 'n_estimators' : [100, 200, 300, 400, 500, 1000] # 디폴트 100 / 1~inf  (inf: 무한대)
# 'learning_rate': [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] 디폴트 0.3/ 0~1 / eta라고 써도 먹힘
# 'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] 디폴트 6 / 0~ inf / 정수
# 'gamma': [0, 1, 2, 3, 4, 5, 7, 10, 100] 디폴트 0/ 0~inf
# 'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10] 디폴트 1 / 0~inf
# 'subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 0/ 0~inf / L1 절대값 가중치 규제 /alpha
# 'reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 1/ 0~inf/ L2 제곱 가중치 규제 /lambda


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
    model = XGBRegressor(**params) 

    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
            #   eval_metric='merror',
              verbose=0,
              early_stopping_rounds=50)

    y_predict = model.predict(x_test)
    results = r2_score(y_test,y_predict)

    return results

lgb_bo = BayesianOptimization(f=lgb_hamsu,
                              pbounds=bayesian_params,
                              random_state=123)

lgb_bo.maximize(init_points=5, n_iter=100)  #초기 2번  ,n-iter : 20번 돌거다! 총 22번 돈다 

print(lgb_bo.max) 

# {'target': 0.8987948988814011, 'params': {'colsample_bylevel': 0.987485671890517, 
#                                           'colsample_bynode': 0.6825865037839414, 'colsample_bytree': 0.6293790245264026, 'gamma': 59.99992779214737, 'max_bin': 
# 365.7889091384681, 'max_depth': 8.976342730790122, 'min_child_weight': 20.58069244248133, 
# 'reg_alpha': 11.321680490624681, 'reg_lambda': 6.533174053608911, 'subsample': 0.410697513862758}}


