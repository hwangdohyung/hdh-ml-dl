import pandas as pd 
import numpy as np 
import tensorflow as tf 
tf.compat.v1.set_random_seed(123)

#1.데이터
path = 'D:\study_data\_data\kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv',index_col =0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

##########전처리############
train_test_data = [train_set, test_set]
sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

print(dataset)

for dataset in train_test_data:
    # 가족수 = 형제자매 + 부모님 + 자녀 + 본인
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    
    # 가족수 > 1이면 동승자 있음
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
embarked_mapping = {'S':0, 'C':1, 'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([\w]+)\.', expand=False)
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].apply(lambda x: 0 if x=="Mr" else 1 if x=="Miss" else 2 if x=="Mrs" else 3 if x=="Master" else 4)

train_set['Cabin'] = train_set['Cabin'].str[:1]
for dataset in train_test_data:
    dataset['Age'].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset['Agebin'] = pd.cut(dataset['Age'], 5, labels=[0,1,2,3,4])
for dataset in train_test_data:
    dataset["Fare"].fillna(dataset.groupby("Pclass")["Fare"].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset['Farebin'] = pd.qcut(dataset['Fare'], 4, labels=[0,1,2,3])
    drop_column = ['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']

for dataset in train_test_data:
    dataset = dataset.drop(drop_column, axis=1, inplace=True)
print(train_set.head())

x_data = train_set.drop(['Survived'], axis=1,)
y_data = train_set['Survived']


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, train_size = 0.8, random_state=123 , stratify = y_data)


x = tf.placeholder(tf.float32, shape=[None, x_data.shape[1]])
y = tf.placeholder(tf.float32, shape=[None])

w1 = tf.Variable(tf.zeros([x_data.shape[1],10])) 
b1 = tf.Variable(tf.zeros([10]))
h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([10,10])) 
b2 = tf.Variable(tf.random_normal([10]))
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

w3 = tf.Variable(tf.random_normal([10,10])) 
b3 = tf.Variable(tf.random_normal([10]))
h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)

w4 = tf.Variable(tf.random_normal([10,10])) 
b4 = tf.Variable(tf.random_normal([10]))
h4 = tf.nn.relu(tf.matmul(h3, w4) + b4)

w5 = tf.Variable(tf.random_normal([10, 1])) 
b5 = tf.Variable(tf.random_normal([1]))
h = tf.nn.sigmoid(tf.matmul(h4, w5) + b5)

# 3-1.컴파일 
loss = -tf.reduce_mean(y*tf.log(h)+(1-y)*tf.log(1-h))               # binary cross entropy
optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001)
train = optimizer.minimize(loss)

# 3-2.훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 1000
for epochs in range(epoch):
    cost_val, hy_val, _  = sess.run([loss, h, train], feed_dict = {x:x_train, y:y_train})
    if epochs %20 == 0:
       print(epochs, 'loss : ',cost_val,'hy : ',hy_val)


# #4.평가, 예측
y_predict = sess.run(tf.cast(sess.run(h,feed_dict={x:x_test}) >= 0.5, dtype=tf.float32))             #텐서를 새로운 형태로 캐스팅하는데 사용한다.부동소수점형에서 정수형으로 바꾼 경우 소수점 버림.
                                                                        #Boolean형태인 경우 True이면 1, False이면 0을 출력한다.

from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score

acc = accuracy_score(y_test,y_predict)
print('acc : ', acc)

sess.close()



