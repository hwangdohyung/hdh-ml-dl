import tensorflow as tf 

x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])

w1 = tf.Variable(tf.random_normal([2, 30], name = 'weights1'))
b1 = tf.Variable(tf.random_normal([30], name = 'bias1'))

h1 = tf.sigmoid(tf.matmul(x, w1) + b1)
#model.add(Dense(30, input_sahpe=(2, ), activation= 'sigmoid'))


