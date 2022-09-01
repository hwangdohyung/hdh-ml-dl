from sklearn.datasets import load_digits
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import to_categorical
from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score

tf.set_random_seed(66)

datasets = load_digits()
x_data = datasets.data 
y_data = datasets.target
y_data = to_categorical(y_data)
x_train,x_test,y_train,y_test = train_test_split(x_data, y_data, train_size=0.9,stratify= y_data,random_state=123)

x = tf.placeholder(tf.float32,shape = [None,x_data.shape[1]])
y = tf.placeholder(tf.float32,shape = [None,y_data.shape[1]])

w1 = tf.Variable(tf.zeros([x_data.shape[1],10]))
b1 = tf.Variable(tf.zeros([10]))
h1 = tf.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([10,8])) 
b2 = tf.Variable(tf.random_normal([8]))
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

w3 = tf.Variable(tf.random_normal([8,6])) 
b3 = tf.Variable(tf.random_normal([6]))
h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)

w4 = tf.Variable(tf.random_normal([6,4])) 
b4 = tf.Variable(tf.random_normal([4]))
h4 = tf.nn.relu(tf.matmul(h3, w4) + b4)

w5 = tf.Variable(tf.random_normal([4,y_data.shape[1]])) 
b5 = tf.Variable(tf.random_normal([y_data.shape[1]]))
h = tf.nn.softmax(tf.matmul(h4, w5) + b5)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h), axis=1))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epoch = 5000
for epochs in range(epoch):
    cost_val, hy_val, _ = sess.run([loss, h, train], feed_dict = {x:x_train, y:y_train})
    if epoch %20 == 0:
        print(epochs, 'loss : ', cost_val)

y_predict = sess.run(tf.argmax(sess.run(h, feed_dict = {x:x_test}),axis=1))
y_test = sess.run(tf.argmax(y_test, axis=1))

acc = accuracy_score(y_predict,y_test)
print("acc : ", acc)

sess.close()

# acc :  0.6277777777777778



