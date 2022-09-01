import tensorflow as tf 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
datasets = load_boston()
x_data, y_data= datasets.data,datasets.target

tf.compat.v1.set_random_seed(123)

y_data = y_data.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, train_size=0.9, random_state=123)

x = tf.placeholder(tf.float32, shape=[None, x_data.shape[1]])
y = tf.placeholder(tf.float32, shape=[None, y_data.shape[1]])

w1 = tf.Variable(tf.random_normal([x_data.shape[1],50])) 
b1 = tf.Variable(tf.zeros([50]))
h1 = tf.matmul(x, w1) + b1

w2 = tf.Variable(tf.random_normal([50,10])) 
b2 = tf.Variable(tf.zeros([10]))
h2 = tf.matmul(h1, w2) + b2

w3 = tf.Variable(tf.random_normal([10,10])) 
b3 = tf.Variable(tf.zeros([10]))
h3 = tf.matmul(h2, w3) + b3

w4 = tf.Variable(tf.random_normal([10,10])) 
b4 = tf.Variable(tf.zeros([10]))
h4 = tf.matmul(h3, w4) + b4

w5 = tf.Variable(tf.random_normal([10,y_data.shape[1]])) 
b5 = tf.Variable(tf.zeros([y_data.shape[1]]))
h =  tf.matmul(h4, w5) + b5

# 3-1.컴파일
loss = tf.reduce_mean(tf.square(h - y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01) 
train = optimizer.minimize(loss)

# 3-2.훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 5000
for epochs in range(epoch):
    cost_val, hy_val, _  = sess.run([loss, h, train],
                                    feed_dict = {x:x_train, y:y_train})
    if epochs %20 == 0:
        print(epochs, 'loss : ', cost_val)

# 4.평가, 예측
from sklearn.metrics import r2_score,mean_absolute_error

y_predict = sess.run(h, feed_dict={x:x_test})

r2 = r2_score(y_test,y_predict)
print('r2 : ', r2)

mae = mean_absolute_error(y_train,hy_val)
print('mae : ', mae)

sess.close()

# r2 :  0.6092823381879977
# mae :  3.494303267801201



