# y = wx + b
import tensorflow as tf 
tf.set_random_seed(123)

#1.데이터
x = [1, 2, 3]
y = [1, 2, 3]

W = tf.Variable(11, dtype = tf.float32)
b = tf.Variable(10, dtype = tf.float32)

#2. 모델구성 
hypothesis = x * W + b  # y = wx + b 

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss) # loss의 최소값을 찾는다.
#model.compile(loss = 'mse', optimizer = 'sgd')

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)             #model.fit
    if step %20 == 0:
        print(step, sess.run(loss), sess.run(W), sess.run(b)) # 출력만 보여주는 것 
        
sess.close()





