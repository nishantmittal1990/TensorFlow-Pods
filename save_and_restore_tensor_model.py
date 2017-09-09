"""
Training a model may take hours, but once you close the tensor session, you may loose all the trained weights and biases.
TensorFlow gives you ability to save your progress using a class called tf.train.saver. This class provides the functionality to save anytf.Variable
to your file system

"""

import tensorflow as tf
# File path the save the data
save_file = './model.ckpt'

# Two tensor variables Weight and bias
weights  = tf.Variable(tf.truncated_normal([2,3]))
bias = tf.Variable(tf.truncated_normal([3]))

# class used to save or/and restore tensor variables
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Weights : ")
    print(sess.run(weights))
    print("Bias :")
    print(sess.run(bias))

    saver.save(sess, save_file)


"""
output:
Weights : 
[[-0.19582979  0.92206436  0.83833575]
 [ 1.62437522  0.45387876 -0.98321545]]
Bias :
[-0.26755854  0.19617774  0.24407507]
"""

"""
The tensor weights and bias are set to random variable using the tf.truncated_normal() function. truncated_normal assign the random value from
normal distribution.
The values are then saved to the file location, "model.ckpt", using the tf.train.Saver.save() function.
ckpt extension stands for checkpoint
"""

"""
LOADING VARIABLES
"""

# Remove the previous weights and bias
tf.reset_default_graph()

# Two Variables weights and bias
weightss = tf.Variable(tf.truncated_normal([2, 3]))
biass = tf.Variable(tf.truncated_normal([3]))

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, save_file)

    print("Weights after load :")
    print(sess.run(weightss))
    print("Bias after load: ")
    print(sess.run(biass))

"""
Output:

Weights after load :
[[-1.72329772 -1.11513615  1.29480493]
 [-0.07041764  0.88065368 -1.44794142]]
Bias after load: 
[ 1.99883735 -0.19326214  1.97306585]

"""

"""
You will notice still we require to create weights and bias tensor in python. The tf.train.Saver.restore() function loads the saved
data into weights and bias.

Since, tf.train.Saver.restore() sets all the Tensorflow variable, we need not to call tf.global_variables_initializer()
"""