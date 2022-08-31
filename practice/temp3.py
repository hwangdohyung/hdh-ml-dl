import pandas as pd
import random
import os
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import KFold,GridSearchCV,train_test_split,RandomizedSearchCV
  
path = 'D:\study_data\_data\dacon_antena/'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정

train_df = pd.read_csv(path + 'train.csv')
test_x = pd.read_csv(path + 'test.csv').drop(columns=['ID'])
train = np.array(train_df)

# print("=============================상관계수 히트 맵==============")
# print(train_df.corr())                    # 상관관계를 확인.  
# import matplotlib.pyplot as plt 
# import seaborn as sns
# sns.set(font_scale=0.3)
# sns.heatmap(data=train_df.corr(),square=True, annot=True, cbar=True) 
# plt.show()

# precent = [0.20,0.40,0.60,0.80]


# print(train_df.describe(percentiles=precent))
# print(train_df.info())  
# print(train_df.columns.values)
# print(train_df.isnull().sum())

#  X_07, X_08, X_09
 
train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature

cols = ["X_10","X_11"]
train_x[cols] = train_x[cols].replace(0, np.nan)

imp = IterativeImputer(estimator = LinearRegression(), 
                       tol= 1e-10, 
                       max_iter=30, 
                       verbose=2, 
                       imputation_order='roman')

train_x = pd.DataFrame(imp.fit_transform(train_x))

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state = 123)

# parameters = {'n_estimators' : [100],
#               'learning_rate': [0.1],
#               'max_depth': [None],
#               'gamma': [1],
#               'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1],
#             #   'subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
#             #   'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
#             #   'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
#             #   'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] ,
#             #   'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10],
#             #   'reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10]
            #   }

model = XGBRegressor(random_state=123,
                   n_estimators = 100,
                   learning_rate = 0.1,
                   max_depth = None,
                   gamma = 1,
                   min_child_weight = 0)

# model = XGBRFRegressor().fit(train_x, train_y)  5

# model = RandomizedSearchCV(xgb, parameters, cv=kfold, n_jobs=8,)
model.fit(train_x,train_y)

print('결과 : ',model.score(train_x,train_y))
 
print(test_x.shape,train_x.shape) 

preds = model.predict(test_x)
# print('결과 : ' , model.score(train_x, train_y))

print('최상의 매개변수 : ', model.best_params_)
print('최상의 점수 : ', model.best_score_)

exit()
submit = pd.read_csv(path + 'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]

submit.to_csv(path + 'submmit.csv', index=False)


