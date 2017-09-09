from batch_helper import batches
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def print_epoch_stats(epoch_i, sess, batch_features,batch_labels):
    """
    Print cost and validation accuracy of each epochs
    :param epoch_i: 
    :param sess: 
    :param batch_features: 
    :param batch_labels: 
    :return: 
    """
    current_cost = sess.run(cost, feed_dict={features: batch_features, labels: batch_labels})
    valid_accuracy = sess.run(accuracy, feed_dict={features: batch_features, labels: batch_labels})
    print('Epochs: {:<4} - Cost: {:<8.3} Valid accuracy: {:<5.3}'.format(epoch_i, current_cost, valid_accuracy))

n_inputs=784 # MINIST input data i.e. 28*28 images
n_classes = 10 #MNIST no of classes i.e. 10

# loading mnist database
mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

# features are already scaled and data is shuffled
# define the MNIST train, validation and test features
train_features = mnist.train.images
valid_features = mnist.validation.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32) # MNIST train labels
valid_labels = mnist.validation.labels.astype(np.float32) # MNIST validation labels
test_labels = mnist.test.labels.astype(np.float32) # MNIST test labels

# features and labels
features = tf.placeholder(tf.float32, [None, n_inputs])
labels = tf.placeholder(tf.float32, [None, n_classes])

#Weights and bias
weights = tf.Variable(tf.random_normal([n_inputs, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

#Logits xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# define loss and optimizer
# Learning rate has been made as place holder because we may want to change the value of learning rate in near future
learning_rate = tf.placeholder(tf.float32)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 128
epochs = 10
learn_rate = 0.001
init = tf.global_variables_initializer()
# call batches function from helper class values passed to batch function are training features and training labels
# This will return features and labels in batches
train_batches = batches(batch_size, train_features, train_labels)

with tf.Session() as sess:
    # run the global variable initializer
    sess.run(init)
    for epoch_i in range(epochs):
        for batch_features, batch_labels in train_batches:
            # the batch features and batch labels returned from batch function are feed to features and labels resp.
            # in the same way learning rate is feed through feed_dict
            train_feed_dict = {features: batch_features, labels: batch_labels, learning_rate: learn_rate}
            # inorder to call the optimizer we require learning rate and cost, where cost intern depends upon logits and labels, and logits in tern depends upon features
            # so we provided features, labels and learning rate as feed_dict
            sess.run(optimizer, feed_dict=train_feed_dict)
        print_epoch_stats(epoch_i, sess, batch_features, batch_labels)
    # similarly in case of test_accuracy we require correct_prediction and correct_prediction requires logits and labels , logits require features
    # so we passes test features and test labels to calculate accuracy
    # there is no need to pass test_features and test_labels to batch method as we need to calculate accuracy, we only need batches in case of training
    # because of which now the value of features has been passed as test_features and value of labels as test_labels
    test_accuracy = sess.run(accuracy, feed_dict={features: test_features, labels: test_labels})

print("Test Accuracy {}".format(test_accuracy))

"""
Output:
D:\Anaconda3\envs\tensorflow\python.exe C:/Users/Intruder/PycharmProjects/Tensor_Flow_Pods/epochs_usingTensor.py
Extracting /datasets/ud730/mnist\train-images-idx3-ubyte.gz
Extracting /datasets/ud730/mnist\train-labels-idx1-ubyte.gz
Extracting /datasets/ud730/mnist\t10k-images-idx3-ubyte.gz
Extracting /datasets/ud730/mnist\t10k-labels-idx1-ubyte.gz
2017-09-09 16:47:14.619455: W C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-09-09 16:47:14.619719: W C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
Epochs: 0    - Cost: 10.0     Valid accuracy: 0.114
Epochs: 1    - Cost: 9.21     Valid accuracy: 0.148
Epochs: 2    - Cost: 8.58     Valid accuracy: 0.159
Epochs: 3    - Cost: 8.06     Valid accuracy: 0.159
Epochs: 4    - Cost: 7.61     Valid accuracy: 0.159
Epochs: 5    - Cost: 7.22     Valid accuracy: 0.159
Epochs: 6    - Cost: 6.86     Valid accuracy: 0.159
Epochs: 7    - Cost: 6.54     Valid accuracy: 0.17 
Epochs: 8    - Cost: 6.24     Valid accuracy: 0.193
Epochs: 9    - Cost: 5.96     Valid accuracy: 0.193
Test Accuracy 0.30880001187324524
"""