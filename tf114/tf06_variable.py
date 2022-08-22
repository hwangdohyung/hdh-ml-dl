import tensorflow as tf 
import numpy as np
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32)
y = tf.Variable([3], dtype=tf.float32)

init = tf.compat.v1.global_variables_initializer()
sess.run(init) #이전의 모든 변수들 초기화 

print(sess.run(x+y))    #[5.]

