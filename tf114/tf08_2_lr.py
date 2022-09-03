# 실습 
# lr 수정해서 epoch를 100번 이하로 줄인다. 
# step = 100이하 , W = 1.99, b = 0.99

x_train_data = [1,2,3]
y_train_data = [3,5,7]


# y = wx + b
import tensorflow as tf 
import numpy as np 
tf.set_random_seed(123)

#1.데이터
x_train = tf.placeholder(tf.float32, shape=[None]) 
y_train = tf.placeholder(tf.float32, shape=[None]) 

W = tf.Variable(tf.random_normal([1],dtype=tf.float32))   
b = tf.Variable(tf.random_normal([1],dtype=tf.float32))    

#2.모델 구성 
h = x_train * W + b  

#3-1. 컴파일 
loss = tf.reduce_mean(tf.square(h - y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.16)
train = optimizer.minimize(loss)

#3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epochs = 100
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, W_val, b_val = sess.run([train, loss, W, b], feed_dict ={x_train:x_train_data, y_train:y_train_data})
        if step %20 == 0:      
            # print(step, sess.run(loss), sess.run(W), sess.run(b))
            print(step, loss_val, W_val, b_val)
    
    x_test_data = [6, 7, 8]
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

    y_predict = x_test * W_val + b_val    # y_predict = model.predict(x_test)

    sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())

    print('[6,7,8] 예측 : ', sess.run(y_predict, feed_dict = {x_test:x_test_data}))








