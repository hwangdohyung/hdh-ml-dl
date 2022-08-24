import tensorflow as tf 
tf.compat.v1.set_random_seed(123)

# 1.데이터 
x1_data = [73., 93., 89., 96., 73.]         # 국어
x2_data = [80., 88., 91., 98., 66.]         # 영어
x3_data = [75., 93., 90., 100., 70.]        # 수학
y_data = [152., 188., 180., 196., 142.]     # 환산점수

x1 = tf.compat.v1.placeholder(tf.float32, shape=None)
x2 = tf.compat.v1.placeholder(tf.float32, shape=None)
x3 = tf.compat.v1.placeholder(tf.float32, shape=None)
y = tf.compat.v1.placeholder(tf.float32, shape=None)

w1 = tf.Variable(tf.random_normal([1],dtype=tf.float32), name ='weights1')
w2 = tf.Variable(tf.random_normal([1],dtype=tf.float32), name ='weights2')
w3 = tf.Variable(tf.random_normal([1],dtype=tf.float32), name ='weights3')
b  = tf.Variable(tf.random_normal([1],dtype=tf.float32), name ='bias')

# 2.모델
h = x1*w1 + x2*w2 + x3*w3 + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(h - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-5)
train = optimizer.minimize(loss)


# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(1000):
     # sess.run(train)
        _, loss_val, w_val1, w_val2, w_val3, b_val = sess.run([train, loss, w1, w2, w3, b], 
                                        feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
        if step %20 == 0:      
           print(step,'\t','loss:', loss_val,'\t','w1:', w_val1, '\t', 'w2:',w_val2,'\t','w3:', w_val3, '\t', 'bias:',b_val)


# 4.평가, 예측
    predict = x1*w_val1 + x2*w_val2 + x3*w_val3 + b 

    y_predict = sess.run(predict, feed_dict = {x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    print('predict :  ', y_predict)


from sklearn.metrics import r2_score,mean_absolute_error
r2 = r2_score(y_data,y_predict)
print('r2 : ', r2)

mae = mean_absolute_error(y_data,y_predict)
print('mae : ', mae)

sess.close()

# predict : [146.22078 189.4696  179.84077 194.27814 149.00502]
# r2 :  0.9600858239347936
# mae :  3.2269866943359373





