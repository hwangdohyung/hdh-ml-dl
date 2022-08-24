import tensorflow as tf 
tf.compat.v1.set_random_seed(123)

# 1. 데이터
x_data = [[73,51,65],[92,98,11],[89,31,33],[99,33,100],[17,66,79]]  # (5,3)
y_data = [[152],[185],[180],[205],[142]]                            # (5,1)


x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])


w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1],dtype=tf.float32), name = 'weights') #행렬곱 shape 을 맞춰줘야함 y(h)가 (5, 1)이므로 w(3,1)을 곱해줘야함
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1],dtype=tf.float32), name='bias')

# 2.모델
h = tf.compat.v1.matmul(x, w) + b  


# 3-1.컴파일
loss = tf.reduce_mean(tf.square(h - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(loss)


# 3-2.훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 2000
for epochs in range(epoch):
    cost_val, hy_val, _  = sess.run([loss, h, train],
                                    feed_dict = {x:x_data, y:y_data})
    if epochs %20 == 0:
        print(epochs, 'loss : ', cost_val, 'weights : ', hy_val)

# 4.평가, 예측
from sklearn.metrics import r2_score,mean_absolute_error
r2 = r2_score(y_data,hy_val)
print('r2 : ', r2)

mae = mean_absolute_error(y_data,hy_val)
print('mae : ', mae)

sess.close()

# r2 :  0.41322196232691455
# mae :  15.57574005126953






