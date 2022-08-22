import tensorflow as tf 
print(tf.__version__)

# print('hello world')

hello = (tf.constant('hello world'))   # constant: 상수(고정값) 
print(hello)

# sess = tf.Session()        # 얘 쓸려면 warning ignore 해주자 
sess = tf.compat.v1.Session()
print(sess.run(hello))

# 텐서플로우1는 출력을 할 때 반드시 sess , run 을 거쳐야한다




