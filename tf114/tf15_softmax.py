import numpy as np 
import tensorflow as tf 
tf.set_random_seed(123)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]     # (8, 4)
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]     

# 2. 모델구성 // 시작
x = tf.placeholder(tf.float32, shape=[None,4])

w = tf.compat.v1.Variable(tf.random_normal([4,3]), name = 'weights') 

b = tf.compat.v1.Variable(tf.random_normal([1,3]), name = 'bias')

y = tf.placeholder(tf.float32, shape=[None,3])


h = tf.nn.softmax(tf.matmul(x, w) + b)

# 3-1.컴파일 
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h), axis=1))     # categorical_crossentropy
train = tf.train.AdamOptimizer(learning_rate = 0.009).minimize(loss)

# 3-2.훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 5000
for epochs in range(epoch):
    cost_val, hy_val, _  = sess.run([loss, h, train],
                                    feed_dict = {x:x_data, y:y_data})
    if epochs %20 == 0:
        print(epochs, 'loss : ', cost_val, 'hy_val : ', hy_val)

# 4.평가, 예측
y_predict = sess.run(tf.argmax(hy_val,axis=1))             #텐서를 새로운 형태로 캐스팅하는데 사용한다.부동소수점형에서 정수형으로 바꾼 경우 소수점 버림.
y_data = sess.run(tf.argmax(y_data,axis=1))             #텐서를 새로운 형태로 캐스팅하는데 사용한다.부동소수점형에서 정수형으로 바꾼 경우 소수점 버림.
                                                                       #Boolean형태인 경우 True이면 1, False이면 0을 출력한다.

from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score

acc = accuracy_score(y_data,y_predict)
print('acc : ', acc)


sess.close()




