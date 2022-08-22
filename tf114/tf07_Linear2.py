# y = wx + b
import tensorflow as tf 
tf.set_random_seed(123)

#1.데이터
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

W = tf.Variable(10, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)

#2.모델 구성 
h = x * W + b 

#3-1. 컴파일 
loss = tf.reduce_mean(tf.square(h - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 2001 
for step in range(epochs):
    sess.run(train)
    print(step, sess.run(loss), sess.run(W), sess.run(b))
        
sess.close()


