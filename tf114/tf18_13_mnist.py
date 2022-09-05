# [실습]
# DNN으로 구성!!
import tensorflow as tf
import numpy as np 
import keras 

tf.compat.v1.set_random_seed(123)

# 1.데이터 
from keras.datasets import mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

x = tf.compat.v1.placeholder(tf.float32, shape = [None,28*28])
y = tf.compat.v1.placeholder(tf.float32, shape = [None,10])

w1 = tf.get_variable('w1', shape= [x_train.shape[1],y_train.shape[1]])  
b1 = tf.get_variable('b1', shape= [y_train.shape[1]])

h = tf.nn.softmax(tf.matmul(x, w1) + b1)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h), axis=1))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 600
for epochs in range(epoch):
    cost_val, hy_val, _  = sess.run([loss, h, train], feed_dict = {x:x_train, y:y_train})
    if epochs %20 == 0:

        print(epochs, 'loss : ', cost_val, 'hy_val : ', hy_val)

# #4.평가, 예측
y_predict = sess.run(h,feed_dict={x:x_test})
y_predict = sess.run(tf.argmax(y_predict,axis=1))         
y_test = sess.run(tf.argmax(y_test,axis=1))             
                                                                    
from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score

acc = accuracy_score(y_test,y_predict)
print('acc : ', acc)

sess.close()

#  acc :  0.9271


