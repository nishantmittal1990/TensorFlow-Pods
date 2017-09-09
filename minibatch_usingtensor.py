from batch_helper import batches
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

leaning_rate = 0.001
n_inputs = 784 #MNIST data input(img shape : 28*28)
n_classes = 10 #MNIST total classes (0-9 digits)

# Load MNIST data
mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

# The features are already scaled and data is shuffled
train_features = mnist.train.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Features and labels
"""
The size of batches would vary, so we need to take advantage of Tensorflow tf.placeholder() function to recieve different batches
if each sample has n_input=784 features and n_classes = 10 possible labels, the dimension for features would be [None, n_input]
and labels would be [None, n_classes]
"""
features = tf.placeholder(tf.float32, [None, n_inputs])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights and Bias
# We also know that during training the model the weights and bias will keep on changing, so, we need to use tf.Variable()
weights = tf.Variable(tf.random_normal([n_inputs, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

#logits - xW+b
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
# We are using softmax cross entropy with logits and labels
# logits the linear equation which is calculated from x*W + b
# x - input features , these features are feed in form of batches and value of same is assigned inside Session call
# similary labels are also feed inform of batches and value of same is feed inside Session call
# both features used inside logits and labels are small batch form of train_features and train_labels
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
# We are using standard gradient descent optimizer and some learning rate and work to minimize the cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=leaning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 128

with tf.Session() as sess:
    # as we are using tf.Variables so we require to initialize global variable initializer
    sess.run(tf.global_variables_initializer())

    # Train optimizer for each batches
    # train features adn train labels is teh collection of MNIST features and labels
    # in order to train our model we will send the same data in form of batches
    for batch_features , batch_labels in batches(batch_size, train_features, train_labels):
        # from the above call we will get features and labels in batches
        # the same will be passes to optimizer in form of feed_dict for features and labels that we received as part of batch process
        sess.run(optimizer, feed_dict={features : batch_features, labels : batch_labels})

    # Test accuracy
    test_accuracy = sess.run(accuracy, feed_dict={features:test_features, labels:test_labels})

print("Test Accuracy {}".format(test_accuracy))


