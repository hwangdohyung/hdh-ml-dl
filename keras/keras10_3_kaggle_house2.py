import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from sklearn.impute import SimpleImputer

path = './_data/kaggle_house/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
train_ID = train['Id']
test_ID = test['Id']
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])

sns.distplot(train['SalePrice'] , fit=norm);

# log1p를 통해 보정된 파라미터 확인
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# 분포 시각화하여서 확인
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')

# QQ-plot 체크를 통한 선형회귀 분석의 가정 확인
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

# 컬럼의 모든 요소에 대하여 log(1+x)로 변환해주는 numy func 중 log1p 사용
train["SalePrice"] = np.log1p(train["SalePrice"])

# 보정된 컬럼의 log 분포 확인 
sns.distplot(train['SalePrice'] , fit=norm);

# log1p를 통해 보정된 파라미터 확인
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# 분포 시각화하여서 확인
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')


# QQ-plot 체크를 통한 선형회귀 분석의 가정 확인
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("전체 데이터셋 shape 확인 : {}".format(all_data.shape))

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
#MSSubClass = 빌딩의 분류를 str데이터 형식으로 변경
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond 를 str형식으로 변경
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#주택이 거래된 년과 월의 경우 str형식으로 변경
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('데이터셋 shape : {}'.format(all_data.shape))

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

train_set = all_data[:len(train)]
test_set = all_data[len(train):]

train_set['SalePrice'] = y_train
############### sale price 다시 드랍 #####################
train_set = train_set.drop(['SalePrice'], axis =1)

# # Check the skew of all numerical features
# skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("\nSkew된 Feautres: \n")
# skewness = pd.DataFrame({'Skew' :skewed_feats})
# skewness.head(10)
# skewness = skewness[abs(skewness) > 0.75]
# print(" {}개의 Skew Numerical Features에 대하여 Box Cox transform 수행".format(skewness.shape[0]))

# from scipy.special import boxcox1p
# skewed_features = skewness.index
# lam = 0.15
# for feat in skewed_features:
#     #all_data[feat] += 1
#     all_data[feat] = boxcox1p(all_data[feat], lam)
    
# #all_data[skewed_features] = np.log1p(all_data[skewed_features])

# all_data = pd.get_dummies(all_data)
# print(all_data.shape)

# train = all_data[:ntrain]
# test = all_data[ntrain:]



x_train, x_test, y_train, y_test = train_test_split(train_set, y_train, train_size=0.8, 
                                            
                                                random_state=77)
print(train_set.shape) 
print(y_train.shape)

#2.모델구성
model = Sequential()
model.add(Dense(100, input_dim=79))
model.add(Dense(100,activation ='relu'))
model.add(Dense(100,activation ='relu'))
model.add(Dense(100,activation ='relu'))
model.add(Dense(1,activation ='relu'))

#3. 컴파일, 훈련
model.compile(loss= 'mae', optimizer ='adam')
model.fit(x_train, y_train, epochs=3000, batch_size=150,verbose=2)

# #4.평가,예측
loss = model.evaluate(train_set, y_train)
print('loss: ', loss)

y_predict = model.predict(x_test).flatten()

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt: 루트 씌우기

y_submmit = model.predict(test_set)
# print(y_submmit)
# print(y_submmit.shape)  


submission = pd.read_csv(path + 'sample_submission.csv')
submission['SalePrice'] = y_submmit

# print(submission)
# print(submission.shape)

submission.to_csv(path + 'submission.csv',index=False)

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

