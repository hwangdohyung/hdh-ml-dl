# m36_dacon_travel.py
import numpy as np
import pandas as pd                               
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score
from tqdm import tqdm_notebook
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import GridSearchCV

#1. 데이터
path = 'D:\study_data\_data\dacon_travle/'
train = pd.read_csv(path + 'train.csv',                 
                        index_col=0)                       

test = pd.read_csv(path + 'test.csv',                                   
                       index_col=0)

sample_submission = pd.read_csv(path + 'sample_submission.csv')


print(train.describe())  # DurationOfPitch, MonthlyIncome
print("=============================상관계수 히트 맵==============")
print(train.corr())                    # 상관관계를 확인.  
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(font_scale=0.3)
sns.heatmap(data=train.corr(),square=True, annot=True, cbar=True) 
# plt.show()



# 결측치를 처리하는 함수를 작성.
def handle_na(data):
    temp = data.copy()
    for col, dtype in temp.dtypes.items():
        if dtype == 'object':
            # 문자형 칼럼의 경우 'Unknown'
            value = 'Unknown'
        elif dtype == int or dtype == float:
            # 수치형 칼럼의 경우 0
            value = 0
        temp.loc[:,col] = temp[col].fillna(value)
    return temp

train_nona = handle_na(train)

# 결측치 처리가 잘 되었는지 확인해 줍니다.
train_nona.isna().sum()

print(train_nona.isna().sum())
object_columns = train_nona.columns[train_nona.dtypes == 'object']
print('object 칼럼 : ', list(object_columns))

# 해당 칼럼만 보아서 봅시다
train_nona[object_columns]

print(train_nona.shape)
print(test.shape)


# LabelEncoder를 준비해줍니다.
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

# LabelEcoder는 학습하는 과정을 필요로 합니다.
encoder.fit(train_nona['TypeofContact'])

#학습된 encoder를 사용하여 문자형 변수를 숫자로 변환해줍니다.
encoder.transform(train_nona['TypeofContact'])
print(train_nona['TypeofContact'])

train_enc = train_nona.copy()

# 모든 문자형 변수에 대해 encoder를 적용합니다.
for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(train_enc[o_col])
    train_enc[o_col] = encoder.transform(train_enc[o_col])

# 결과를 확인합니다.
print(train_enc)
# 결측치 처리
test = handle_na(test)

# 문자형 변수 전처리
for o_col in object_columns:
    encoder = LabelEncoder()
    
    # test 데이터를 이용해 encoder를 학습하는 것은 Data Leakage 입니다! 조심!
    encoder.fit(train_nona[o_col])
    
    # test 데이터는 오로지 transform 에서만 사용되어야 합니다.
    test[o_col] = encoder.transform(test[o_col])

# 결과를 확인
print(test)


# 모델 선언
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from xgboost import XGBClassifier, XGBRegressor 
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# import matplotlib.pyplot as plt

# train_enc.plot.box()
# plt.title('boston')
# plt.xlabel('Feature')
# plt.ylabel('data')
# plt.show()

# exit()

# 분석할 의미가 없는 칼럼을 제거합니다.
# 상관계수 그래프를 통해 연관성이 적은것과 - 인것을 빼준다.
train = train_enc.drop(columns=['TypeofContact','NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar', 'MonthlyIncome'])  
test = test.drop(columns=['TypeofContact','NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar', 'MonthlyIncome'])
# 'TypeofContact','NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar', 'MonthlyIncome'

# 학습에 사용할 정보와 예측하고자 하는 정보를 분리합니다.
x = train.drop(columns=['ProdTaken'])
y = train[['ProdTaken']]

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state=72, train_size=0.88,shuffle=True,stratify=y)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split, KFold , StratifiedKFold
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# 모델 학습
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, gamma = 1, subsample=1, colsample_bytree = 1, max_depth=4,random_state=123)


# ##########################GridSearchCV###############################
# n_splits = 5

# parameters = {'n_estimators':[1000],
#               'learning_rate':[0.1],
#               'max_depth':[3],
#               'gamma': [0],
#             #   'min_child_weight':[1],
#               'subsample':[1],
#               'colsample_bytree':[1],
#             #   'colsample_bylevel':[1],
#             #   'colsample_byload':[1],
#             #   'reg_alpha':[0],
#             #   'reg_lambda':[1]
#               }  

# kfold = KFold(n_splits=n_splits ,shuffle=True, random_state=123)
# xgb = XGBClassifier(random_state=123,
#                     )

# model = GridSearchCV(xgb,param_grid=parameters, cv =kfold, n_jobs=8)
##########################GridSearchCV###############################


model = RandomForestClassifier()

model.fit(x_train,y_train)

prediction = model.predict(x_test)
prediction1 = model.predict(test)

print('----------------------예측된 데이터의 상위 10개의 값 확인--------------------\n')

print('acc : ', accuracy_score(prediction,y_test))

print(prediction[:10])
# print(model.score(x_train, y_train))
# 예측된 값을 정답파일과 병합
print(prediction.shape)

sample_submission['ProdTaken'] = prediction1

# 정답파일 데이터프레임 확인
print(sample_submission)

sample_submission.to_csv(path+'sample_submission0820_3.csv',index = False)