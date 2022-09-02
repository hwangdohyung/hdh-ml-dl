import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from xgboost import XGBClassifier,XGBRegressor
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


### 1.데이터 ###
path = 'D:\study_data\_data\dacon_travle/'

train = pd.read_csv( path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
submit = pd.read_csv(path + 'sample_submission.csv',index_col=0)

print(train.shape, test.shape) # (1955, 19) (2933, 18)


##################### 라벨인코더 ######################
le = LabelEncoder()

idxarr = train.columns
idxarr = np.array(idxarr)

for i in idxarr:
      if train[i].dtype == 'object':
        train[i] = le.fit_transform(train[i])
        test[i] = le.fit_transform(test[i])


# ### 상관관계 ###
# sns.set(font_scale= 0.8 )
# sns.heatmap(data=train.corr(), square= True, annot=True, cbar=True) # square: 정사각형, annot: 안에 수치들 ,cbar: 옆에 bar

# plt.show() 
# train.to_csv(path + 'train22.csv',index=False)

### 결측치 ###
#트레인,테스트 합치기 #
alldata = pd.concat((train, test), axis=0)
alldata_index = alldata.index

# # 아웃라이어를 찾아서 NaN으로 바꾸기
# def remove_outlier(input_data):
#     q1 = input_data.quantile(0.25) # 제 1사분위수
#     q3 = input_data.quantile(0.75) # 제 3사분위수
#     iqr = q3 - q1 # IQR(Interquartile range) 계산
#     minimum = q1 - (iqr * 1.5) # IQR 최솟값
#     maximum = q3 + (iqr * 1.5) # IQR 최댓값
#     # IQR 범위 내에 있는 데이터만 산출(IQR 범위 밖의 데이터는 이상치)
#     df_removed_outlier = input_data[(minimum < input_data) & (input_data < maximum)]
#     return df_removed_outlier

# datasets = remove_outlier(alldata)

# # # 이상치를 interpolate로 채우기
# alldata = datasets.interpolate()


alldata = alldata.drop(['Age','MonthlyIncome'],axis=1)

print(alldata.shape)
train = alldata[:len(train)]
test = alldata[len(train):]

train = train.dropna()

mean = test['DurationOfPitch'].mean()
test['DurationOfPitch'] = test['DurationOfPitch'].fillna(mean)

mean = test['NumberOfFollowups'].mean()
test['NumberOfFollowups'] = test['NumberOfFollowups'].fillna(mean)

mean = test['PreferredPropertyStar'].median()
test['PreferredPropertyStar'] = test['PreferredPropertyStar'].fillna(mean)

mean = test['NumberOfTrips'].median()
test['NumberOfTrips'] = test['NumberOfTrips'].fillna(mean)

mean = test['NumberOfChildrenVisiting'].median()
test['NumberOfChildrenVisiting'] = test['NumberOfChildrenVisiting'].fillna(mean)

print(test.isnull().sum())

x = train.drop('ProdTaken',axis=1)
print(x.shape)      #(1955, 18)
y = train['ProdTaken']
print(y.shape)      #(1955,)
print(submit.shape) #(2933, 1)
print(test.columns)

test = test.drop('ProdTaken',axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, 
                                                     random_state=123,stratify=y)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state = 123)

parameters = {'n_estimators' : [200],
              'learning_rate': [0.2],
              'max_depth': [10],
              'gamma': [0],
              'min_child_weight': [0.1],
              'subsample': [1],
              'colsample_bytree': [1],
              'colsample_bylevel': [0.5],
              'colsample_bynode': [1] ,
              'reg_alpha': [0.1],
              'reg_lambda':[1]
              }
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel


model = XGBClassifier(random_state = 123,
                                    n_estimators =200,
              learning_rate= 0.2,
              max_depth= 10,
              gamma=0,
              min_child_weight= 0.1,
              subsample= 1,
              colsample_bytree= 1,
              colsample_bylevel= 0.5,
              colsample_bynode= 1 ,
              reg_alpha= 0.1,
              reg_lambda=1)
# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)
model.fit(x_train,y_train)


# print(model.feature_importances_)

# thresholds = model.feature_importances_
# print('================================')
# for thresh in thresholds:
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)  #prefit=true: feature importance 자기보다 같거나 큰놈들을 전부 반환해준다
    
#     select_x_train = selection.transform(x_train)
#     select_x_test= selection.transform(x_test)
#     print(select_x_train.shape,select_x_test.shape)

#     selection_model = XGBClassifier(random_state = 123,
#                                     n_estimators = 100,
#                                     learning_rate = 0.1,
#                                     max_depth = 3,
#                                     gamma = 1, 
#                                     )
#     selection_model.fit(select_x_train, y_train)
    
#     y_predict = selection_model.predict(select_x_test)
#     score = accuracy_score(y_test, y_predict)
    
#     print("Thres=%.3f, n=%d, acc: %.2f%%"
#           %(thresh, select_x_train.shape[1], score*100))

#2. 모델 
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline 
from xgboost import XGBRegressor
# print('최상의 매개변수 : ', model.best_params_)
# print('최상의 점수 : ', model.best_score_)


submission = pd.read_csv(path + 'sample_submission.csv')
submission['ProdTaken'] = submit

submission.to_csv(path +'last2.csv',index=False)


