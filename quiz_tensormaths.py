"""
Let's apply what you learned to convert an algorithm to TensorFlow. 
The code below is a simple algorithm using division and subtraction. 
Convert the following algorithm in regular Python to TensorFlow and print the results of the session. 
You can use tf.constant() for the values 10, 2, and 1.

"""
# Solution is available in the other "solution.py" tab
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
#x = 10
x = tf.placeholder(tf.float32)
#y = 2
y = tf.placeholder(tf.float32)
#divide_x_y = tf.divide(x, y)
#z = x/y - 1

# TODO: Print z from a session
with tf.Session() as sess:
    z = sess.run(tf.subtract(tf.divide(x,y), 1), feed_dict={x:10,y:2})
    print(z)