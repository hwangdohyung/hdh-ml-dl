import tensorflow as tf 
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)


#실습 덧셈 node3 뺄셈 node4 곱셈 node5 나눗셈 node6
#만들기 node3 = node1 + node2
node3 = tf.add(node1,node2)

# node4 = node2 - node1
node4 = tf.subtract(node2,node1)

# node5 = node1 * node2
# node5 = tf.matmul(node2,node1) # 행렬곱
node5 = tf.multiply(node2,node1)


# node6 = node2 / node1
# node6 = tf.mod(node2,node1)   # 나머지 구하는것
node6 = tf.divide(node2,node1)  # 몫


sess = tf.compat.v1.Session()
print(sess.run(node3))          # 5.0
print(sess.run(node4))          # 1.0
print(sess.run(node5))          # 6.0
print(sess.run(node6))          # 1.5



