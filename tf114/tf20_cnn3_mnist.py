import tensorflow as tf 
import keras 
import numpy as np 

tf.compat.v1.set_random_seed(123)

#즉시실행모드
tf.compat.v1.disable_eager_execution()

# 1.데이터 
from keras.datasets import mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

# 2.모델구성
x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])  # input_shape
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

# Layer1 

w1 = tf.compat.v1.get_variable('w1', shape= [2, 2, 1, 32]) #  2, 2=kenel size,  1=컬러,  64=output node(filter)
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding= 'SAME') # 가운데 2개 스트라이드 , 양옆에 1,1 은 차원 맞춰주는 것
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides= [1,2,2,1], padding= 'SAME')

# Layer2 

w2 = tf.compat.v1.get_variable('w2', shape= [3, 3, 32, 16]) 
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding= 'VALID') 
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool2d(L2, ksize=[1,2,2,1], strides= [1,2,2,1], padding= 'SAME')

print(L2)

# Layer3

w3 = tf.compat.v1.get_variable('w3', shape= [3, 3, 16, 8]) 
L3 = tf.nn.conv2d(L2, w3, strides=[1,1,1,1], padding= 'VALID') 
L3 = tf.nn.elu(L3)

print(L3)

# Flatten
L_flat = tf.reshape(L3, [-1, 4*4*8])
print("FLATTEN : ", L_flat)

# Layer4 DNN
w4 = tf.compat.v1.get_variable('w4', shape=[4*4*8, 30], 
                            #    initializer=tf.compat.v1.layers.xavier_initializer()
                               )
b4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([30], name= 'b4'))
L4 = tf.nn.selu(tf.matmul(L_flat, w4) + b4)
L4 = tf.nn.dropout(L4,rate = 0.3,
                #    keep_prob =0.7
                ) 

# Layer5 DNN

w5 = tf.compat.v1.get_variable('w5', shape=[30, 10], 
                            #    initializer=tf.contrib.layers.xavier_initializer()
                               )
b5 = tf.Variable(tf.compat.v1.random_normal([10], name= 'b5'))
L5 = tf.matmul(L4, w5) + b5
h = tf.nn.softmax(L5)


# loss = tf.compat.v1.reduce_mean(-tf.reduce_sum(y * tf.log(h), axis=1))     # categorical_crossentropy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= h, labels=y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 30
batch_size = 100
total_batch = int(len(x_train)/batch_size)  # 60000/100 = 600

for epochs in range(epoch):         # 총 30번 돈다
    avg_loss = 0
    for i in range(total_batch):   # 총 600번돈다
        start = i * batch_size     # 0
        end = start + batch_size   # 100   
        batch_x, batch_y = x_train[start:end], y_train[start:end]      # 0~100
       
        feed_dict = {x:batch_x, y:batch_y}
        batch_loss, _ = sess.run([loss, optimizer],feed_dict=feed_dict) 
        
        avg_loss += batch_loss / total_batch
        
    print('Epoch : ',  '%04d' %(epoch + 1), 'loss : {:.9f}'.format(avg_loss))
print('훈련 끝!!')    

prediction = tf.equal(tf.argmax(h,1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('acc: ', sess.run(accuracy, feed_dict = {x:x_test, y:y_test}))


# # #4.평가, 예측
# y_predict = sess.run(h,feed_dict={x:x_test})
# y_predict = sess.run(tf.argmax(y_predict,axis=1))           #텐서를 새로운 형태로 캐스팅하는데 사용한다.부동소수점형에서 정수형으로 바꾼 경우 소수점 버림.
# y_test = sess.run(tf.argmax(y_test,axis=1))             #텐서를 새로운 형태로 캐스팅하는데 사용한다.부동소수점형에서 정수형으로 바꾼 경우 소수점 버림.
#                                                                        #Boolean형태인 경우 True이면 1, False이면 0을 출력한다.

# from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score

# acc = accuracy_score(y_test,y_predict)
# print('acc : ', acc)

# sess.close()

# acc:  0.8082

