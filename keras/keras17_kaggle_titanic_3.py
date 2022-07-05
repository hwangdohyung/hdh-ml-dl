#[실습]#
import numpy as np 
import pandas as pd
from sklearn import metrics
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


# datasets.describe()
# datasets.info()
# datasets.isnull().sum()
# pandas의 y라벨의 종류가 무엇인지 확인하는 함수 쓸것          # unique() , value counts(ascending= Ture) *ascending Ture: 오름차순
# numpy에서는 np.unique(y, return_counts=True)

#1.데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv',index_col =0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

# #########전처리############
#sex컬럼
train_set['Sex_clean'] = train_set['Sex'].astype('category').cat.codes
test_set['Sex_clean'] = test_set['Sex'].astype('category').cat.codes
#emnarked
train_set['Embarked'].fillna('S', inplace=True)
train_set['Embarked_clean'] = train_set['Embarked'].astype('category').cat.codes
test_set['Embarked_clean'] = test_set['Embarked'].astype('category').cat.codes
# sibsp,parch
train_set['Family'] = 1 + train_set['SibSp'] + train_set['Parch']
test_set['Family'] = 1 + test_set['SibSp'] + test_set['Parch']
train_set['Solo'] = (train_set['Family'] == 1)
test_set['Solo'] = (test_set['Family'] == 1)
#fare
train_set['FareBin'] = pd.qcut(train_set['Fare'], 5)
test_set['FareBin'] = pd.qcut(test_set['Fare'], 5)
train_set['Fare_clean'] = train_set['FareBin'].astype('category').cat.codes
test_set['Fare_clean'] = test_set['FareBin'].astype('category').cat.codes
#name
train_set['Title'] = train_set['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_set['Title'] = test_set['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train_set['Title'] = train_set['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
train_set['Title'] = train_set['Title'].replace('Mlle', 'Miss')
train_set['Title'] = train_set['Title'].replace('Ms', 'Miss')
train_set['Title'] = train_set['Title'].replace('Mme', 'Mrs')
test_set['Title'] = test_set['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
test_set['Title'] = test_set['Title'].replace('Mlle', 'Miss')
test_set['Title'] = test_set['Title'].replace('Ms', 'Miss')
test_set['Title'] = test_set['Title'].replace('Mme', 'Mrs')
train_set['Title_clean'] = train_set['Title'].astype('category').cat.codes
test_set['Title_clean'] = test_set['Title'].astype('category').cat.codes
#age
train_set["Age"].fillna(train_set.groupby("Title")["Age"].transform("median"), inplace=True)
test_set["Age"].fillna(test_set.groupby("Title")["Age"].transform("median"), inplace=True)
train_set.loc[ train_set['Age'] <= 10, 'Age_clean'] = 0
train_set.loc[(train_set['Age'] > 10) & (train_set['Age'] <= 16), 'Age_clean'] = 1
train_set.loc[(train_set['Age'] > 16) & (train_set['Age'] <= 20), 'Age_clean'] = 2
train_set.loc[(train_set['Age'] > 20) & (train_set['Age'] <= 26), 'Age_clean'] = 3
train_set.loc[(train_set['Age'] > 26) & (train_set['Age'] <= 30), 'Age_clean'] = 4
train_set.loc[(train_set['Age'] > 30) & (train_set['Age'] <= 36), 'Age_clean'] = 5
train_set.loc[(train_set['Age'] > 36) & (train_set['Age'] <= 40), 'Age_clean'] = 6
train_set.loc[(train_set['Age'] > 40) & (train_set['Age'] <= 46), 'Age_clean'] = 7
train_set.loc[(train_set['Age'] > 46) & (train_set['Age'] <= 50), 'Age_clean'] = 8
train_set.loc[(train_set['Age'] > 50) & (train_set['Age'] <= 60), 'Age_clean'] = 9
train_set.loc[ train_set['Age'] > 60, 'Age_clean'] = 10

test_set.loc[ test_set['Age'] <= 10, 'Age_clean'] = 0
test_set.loc[(test_set['Age'] > 10) & (test_set['Age'] <= 16), 'Age_clean'] = 1
test_set.loc[(test_set['Age'] > 16) & (test_set['Age'] <= 20), 'Age_clean'] = 2
test_set.loc[(test_set['Age'] > 20) & (test_set['Age'] <= 26), 'Age_clean'] = 3
test_set.loc[(test_set['Age'] > 26) & (test_set['Age'] <= 30), 'Age_clean'] = 4
test_set.loc[(test_set['Age'] > 30) & (test_set['Age'] <= 36), 'Age_clean'] = 5
test_set.loc[(test_set['Age'] > 36) & (test_set['Age'] <= 40), 'Age_clean'] = 6
test_set.loc[(test_set['Age'] > 40) & (test_set['Age'] <= 46), 'Age_clean'] = 7
test_set.loc[(test_set['Age'] > 46) & (test_set['Age'] <= 50), 'Age_clean'] = 8
test_set.loc[(test_set['Age'] > 50) & (test_set['Age'] <= 60), 'Age_clean'] = 9
test_set.loc[ test_set['Age'] > 60, 'Age_clean'] = 10
#cabin
mapping = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'T': 7
}
train_set['Cabin_clean'] = train_set['Cabin'].str[:1]
train_set['Cabin_clean'] = train_set['Cabin_clean'].map(mapping)
train_set['Cabin_clean'] = train_set.groupby('Pclass')['Cabin_clean'].transform('median')

test_set['Cabin_clean'] = test_set['Cabin'].str[:1]
test_set['Cabin_clean'] = test_set['Cabin_clean'].map(mapping)
test_set['Cabin_clean'] = test_set.groupby('Pclass')['Cabin_clean'].transform('median')

train_set= train_set.drop(['Name','Sex','Cabin','Fare','Embarked','FareBin','Fare','Ticket','Parch','SibSp','Age','Solo','Title'],axis=1)
test_set = test_set.drop(['Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Solo','FareBin','Title','Embarked'],axis=1)
print(train_set)
print(test_set)


x = train_set.drop(['Survived'], axis=1,)
y = train_set['Survived']


#2.모델
model = Sequential()
model.add(Dense(30, input_dim=8))
model.add(Dense(30,activation ='relu'))
model.add(Dense(30,activation ='relu'))
model.add(Dense(30,activation ='relu'))
model.add(Dense(30,activation ='relu'))
model.add(Dense(1,activation = 'sigmoid'))


#.컴파일,훈련
model.compile(loss= 'binary_crossentropy',optimizer='adam')
earlyStopping= EarlyStopping(monitor= 'val_loss',patience=50,mode='min',restore_best_weights=True,verbose=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size= 0.3,random_state=77)
model.fit(x_train, y_train, epochs=1000, batch_size=32,validation_split=0.2,callbacks=earlyStopping, verbose=1)



#4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)



y_predict = model.predict(x_test)

y_predict = y_predict.round() 

acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)

#제출
y_submmit = model.predict(test_set)

y_submmit = y_submmit.round()
y_submmit = y_submmit.astype('int32')
 

submission = pd.read_csv(path + 'gender_submission.csv')
submission['Survived'] = y_submmit

submission.to_csv(path + 'submission.csv',index=False)


