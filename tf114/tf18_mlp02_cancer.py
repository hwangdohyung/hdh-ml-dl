import tensorflow as tf 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
import numpy as np 
tf.set_random_seed(66)

# 1.데이터
datasets = load_breast_cancer()

x_data = datasets.data
y_data = datasets.target 
y_data = y_data.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, train_size = 0.9, random_state=123, stratify = y_data)

x = tf.placeholder(tf.float32, shape=[None, x_data.shape[1]])
y = tf.placeholder(tf.float32, shape=[None, y_data.shape[1]])

w1 = tf.Variable(tf.zeros([x_data.shape[1],50])) 
b1 = tf.Variable(tf.zeros([50]))
h1 = tf.matmul(x, w1) + b1

w2 = tf.Variable(tf.random_normal([50,10])) 
b2 = tf.Variable(tf.random_normal([10]))
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

w3 = tf.Variable(tf.random_normal([10,10])) 
b3 = tf.Variable(tf.random_normal([10]))
h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)

w4 = tf.Variable(tf.random_normal([10,10])) 
b4 = tf.Variable(tf.random_normal([10]))
h4 = tf.nn.relu(tf.matmul(h3, w4) + b4)

w5 = tf.Variable(tf.random_normal([10,y_data.shape[1]])) 
b5 = tf.Variable(tf.random_normal([y_data.shape[1]]))
h = tf.nn.sigmoid(tf.matmul(h4, w5) + b5)

# 3-1.컴파일 
loss = -tf.reduce_mean(y*tf.log(h)+(1-y)*tf.log(1-h))               # binary cross entropy
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0000001)
train = optimizer.minimize(loss)

# 3-2.훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 5000
for epochs in range(epoch):
    cost_val, hy_val, _  = sess.run([loss, h, train], feed_dict = {x:x_train, y:y_train})
    if epochs %20 == 0:
       print(epochs, 'loss : ',cost_val)

# #4.평가, 예측
y_predict = sess.run(tf.cast(sess.run(h,feed_dict={x:x_test})>0.5, dtype=tf.float32))             #텐서를 새로운 형태로 캐스팅하는데 사용한다.부동소수점형에서 정수형으로 바꾼 경우 소수점 버림.
                                                                        #Boolean형태인 경우 True이면 1, False이면 0을 출력한다.

from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score

acc = accuracy_score(y_test,y_predict)
print('acc : ', acc)


sess.close()

# 4980 loss :  0.26280499
# acc :  0.9298245614035088


