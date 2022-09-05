import tensorflow as tf 
import numpy as np 
tf.compat.v1.set_random_seed(123)

# 1.데이터
x_train = np.array([[[[1],[2],[3]],
                     [[4],[5],[6]],
                     [[7],[8],[9]]]])

print(x_train.shape)  # (1, 3, 3, 1)

x = tf.compat.v1.placeholder(tf.float32, [None, 3, 3, 1])

w = tf.compat.v1.constant([[[[1.]],[[1.]]],
                 [[[1.]],[[1.]]]])
print(w)

l1 = tf.nn.conv2d(x, w, strides=(1,2,2,1), padding= 'SAME')
print(l1)

sess = tf.compat.v1.Session()
output = sess.run(l1, feed_dict= {x:x_train})

print(output)

print(output.shape) 



