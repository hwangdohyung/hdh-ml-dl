# y = wx + b
import tensorflow as tf 
tf.set_random_seed(123)

#1.데이터
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

W = tf.Variable(11, dtype=tf.float32)
b = tf.Variable(10, dtype=tf.float32)

#2.모델 구성 
h = x * W + b   # 실제 연산방법

#3-1. 컴파일 
loss = tf.reduce_mean(tf.square(h - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련
with tf.compat.v1.Session() as sess:
# sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 2001 
    for step in range(epochs):
        sess.run(train)
        if step %20 == 0:       # %: 나머지를 구하는것 
            print(step, sess.run(loss), sess.run(W), sess.run(b))
        
# sess.close()




