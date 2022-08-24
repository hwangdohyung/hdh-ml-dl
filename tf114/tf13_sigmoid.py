import tensorflow as tf
tf.compat.v1.set_random_seed(123)

# 1.데이터
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]] # (6, 2)
y_data = [[0], [0], [0], [1], [1], [1]]             # (6, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]), name = 'weights') 
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')

# 2.모델
h = tf.sigmoid(tf.matmul(x, w) + b)  
# == model.add(Dense(1, activation = 'sigmoid', input_dim =2))

# 3-1.컴파일 
loss = -tf.reduce_mean(y*tf.log(h)+(1-y)*tf.log(1-h))               # binary cross entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
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
y_predict = sess.run(tf.cast(hy_val>0.5, dtype=tf.float32))             #텐서를 새로운 형태로 캐스팅하는데 사용한다.부동소수점형에서 정수형으로 바꾼 경우 소수점 버림.
                                                                        #Boolean형태인 경우 True이면 1, False이면 0을 출력한다.

from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score

acc = accuracy_score(y_data,y_predict)
print('acc : ', acc)

mae = mean_absolute_error(y_data,hy_val)
print('mae : ', mae)

sess.close()

# acc :  1.0
# mae :  0.20064415037631989





