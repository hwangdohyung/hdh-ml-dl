from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, -1) / 255  # (60000, 784)
x_test = x_test.reshape(10000, -1) / 255  # (10000, 784)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_train.shape[1]])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

w = tf.compat.v1.Variable(tf.random.normal([x_train.shape[1], 64], mean=0.0, stddev=tf.math.sqrt(2/(784+64)), name='weight'))
b = tf.compat.v1.Variable(tf.zeros([64]), name='bias')  

#2. 모델구성
hidden_layer1 = tf.matmul(x, w) + b

w1 = tf.compat.v1.Variable(tf.random.normal([64, 32], mean=0, stddev=tf.math.sqrt(2/(64+32)), name='weight1'))
b1 = tf.compat.v1.Variable(tf.zeros([32], name='bias1'))

hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.normal([32, 16], mean=0, stddev=tf.math.sqrt(2/(32+16)), name='weight2'))
b2 = tf.compat.v1.Variable(tf.zeros([16], name='bias2'))

hidden_layer3 = tf.nn.relu(tf.matmul(hidden_layer2, w2) + b2)

w3 = tf.compat.v1.Variable(tf.random.normal([16, 4], mean=0, stddev=tf.math.sqrt(2/(16+4)), name='weight3'))
b3 = tf.compat.v1.Variable(tf.zeros([4], name='bias3'))

hidden_layer4 = tf.nn.relu(tf.matmul(hidden_layer3, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random.normal([4, 10], mean=0, stddev=tf.math.sqrt(2/(4+1)), name='weight3'))
b4 = tf.compat.v1.Variable(tf.zeros([10], name='bias3'))

output = tf.nn.softmax(tf.matmul(hidden_layer4, w4) + b4)

# 모델 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(output), axis=1)) # categorical_crossentropy

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0024).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(201):
        _, w_val, = sess.run([optimizer, loss], feed_dict = {x:x_train,y:y_train})
        if step % 10 ==0:
            print(step, w_val)

    y_acc_test = sess.run(tf.argmax(y_test, 1))
    predict = sess.run(tf.argmax(sess.run(output, feed_dict={x:x_test}), 1))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y_acc_test),dtype=tf.float32))
    a = sess.run(accuracy,feed_dict = {x:x_test,y:y_test})
    print("\nacc : ", a)