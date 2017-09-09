"""
To create the cross entropy in tensor. we need to use two functions:
tf.reduce_sum()
tf.log()
Cross_entropy = -Sigmoid(yj*ln(y_cap j))

x = tf.reduce_sum([2,3,4,5,6]) --> function takes array of numbers and sum them together
x = tf.log(100) --> takes natural log of a number
"""

import tensorflow as tf
softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

soft_max = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(soft_max)))

with tf.Session() as sess:
    print(sess.run(cross_entropy, feed_dict={one_hot:one_hot_data, soft_max:softmax_data}))