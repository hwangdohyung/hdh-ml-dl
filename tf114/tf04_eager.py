import tensorflow as tf 
print(tf.__version__)
print(tf.executing_eagerly())  #False

#즉시실행모드!!!(tensor2 를 쓰는데 1문법을 실행시키고 싶을 때)
# tf.compat.v1.disable_eager_execution() # 즉시실행 모드를 끄겠다

# print(tf.executing_eagerly()) 


hello = tf.constant('Hello World')

sess = tf.compat.v1.Session()

print(sess.run(hello))


