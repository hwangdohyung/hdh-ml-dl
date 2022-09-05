import tensorflow as tf 
import keras 
import numpy as np 

tf.compat.v1.set_random_seed(123)

# 1.데이터 
from keras.datasets import mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

# 2.모델구성
x = tf.placeholder(tf.float32, [None, 28, 28, 1])  # input_shape
y = tf.placeholder(tf.float32, [None, 10])

w1 = tf.get_variable('w1', shape= [2, 2, 1, 64]) #  2, 2=kenel size,  1=컬러,  64=output node(filter)
# w2 = tf.get_variable('w2', shape= [2, 2, 64, 64])

L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding= 'VALID') # 가운데 2개 스트라이드 , 양옆에 1,1 은 차원 맞춰주는 것

print(w1) # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1) # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)





