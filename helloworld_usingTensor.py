import tensorflow as tf

hello_string = tf.constant('Hello World !!')

with tf.Session() as sess:
    output = sess.run(hello_string)
    print(output)


# Same can be done by using place holder and session free_dict. Place holder is used in a place where we don't want to user constant

hello_string1 = tf.placeholder(tf.string)
with tf.Session() as sesss:
    output1 = sesss.run(hello_string1, feed_dict={hello_string1:'Hello Nishant!!!'})
    print(output1)

"""
Note: if the type defined in placeholder doesn't matches with feed_dict type then we will get "Value error: invalid literal for..."
"""