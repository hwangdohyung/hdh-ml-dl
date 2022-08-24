import numpy as np 
import pandas as pd 
import tensorflow as tf 
from sklearn.model_selection import train_test_split

#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv') 
            
test_set = pd.read_csv(path + 'test.csv') 


train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) 
train_set.drop('casual',axis=1,inplace=True)
train_set.drop('registered',axis=1,inplace=True)
test_set.drop('datetime',axis=1,inplace=True) 


x_data = train_set.drop(['count'], axis=1)  
y_data = train_set['count'] 

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, train_size = 0.8, random_state=123)

# print(x_train.dtype,y_train.dtype)

x = tf.placeholder(tf.float32, shape=[None, x_data.shape[1]])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([x_data.shape[1],1]), name = 'weights') 
b = tf.Variable(tf.random_normal([1]), name='bias')

# 2.모델
h = tf.compat.v1.matmul(x, w) + b  

# 3-1.컴파일
loss = tf.reduce_mean(tf.square(h - y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.1) 
train = optimizer.minimize(loss)

# 3-2.훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 1000
for epochs in range(epoch):
    cost_val, hy_val, _  = sess.run([loss, h, train],
                                    feed_dict = {x:x_train, y:y_train})
    if epochs %20 == 0:
        print(epochs, 'loss : ', cost_val)

# 4.평가, 예측
from sklearn.metrics import r2_score,mean_absolute_error

y_predict = sess.run(h, feed_dict={x:x_test})

r2 = r2_score(y_test,y_predict)
print('r2 : ', r2)

mae = mean_absolute_error(y_train,hy_val)
print('mae : ', mae)

sess.close()




