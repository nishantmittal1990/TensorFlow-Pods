"""
Let's see how to train a model and save the weights
"""

# let's start with model

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

leaning_rate = 0.001
n_input = 784
n_classes = 10

# import MNIST
mnist = input_data.read_data_sets(".", one_hot=True)

# Feature and labels
features = tf.placeholder(np.float32, [None, n_input])
labels = tf.placeholder(np.float32, [None, n_classes])

# weights and bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# logits
logits = tf.add(tf.matmul(features, weights), bias)

# cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=leaning_rate).minimize(cost)

# calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# let's train the model and save the weights

save_file = './train_model.ckpt'

batch_size = 128
n_epochs = 100

saver = tf.train.Saver()
import math
# launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # training cycle
    for epoch in range(n_epochs):
        total_batch = math.ceil(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_feature, batch_label = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={features:batch_feature, labels:batch_label})

        if epoch % 10 == 0:
            valid_accracy = sess.run(accuracy, feed_dict={features:mnist.validation.images, labels:mnist.validation.labels})
            print('Epoch {:<3} - Validation Accuracy : {}'.format(epoch, valid_accracy))

    saver.save(sess, save_file)
    print("Trained model saved")

"""
******************** Load Trained model ***********************************
"""

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:

    saver.restore(sess, save_file)
    test_accuracy = sess.run(accuracy, feed_dict={features:mnist.test.images, labels:mnist.test.labels})

print("Test Accuracy : {}".format(test_accuracy))

"""
Output:
D:\Anaconda3\envs\tensorflow\python.exe C:/Users/Intruder/PycharmProjects/Tensor_Flow_Pods/save_weights_for_trained_model.py
Extracting .\train-images-idx3-ubyte.gz
Extracting .\train-labels-idx1-ubyte.gz
Extracting .\t10k-images-idx3-ubyte.gz
Extracting .\t10k-labels-idx1-ubyte.gz
2017-09-09 22:23:40.064453: W C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-09-09 22:23:40.065176: W C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
Epoch 0   - Validation Accuracy : 0.07259999960660934
Epoch 10  - Validation Accuracy : 0.27480000257492065
Epoch 20  - Validation Accuracy : 0.43059998750686646
Epoch 30  - Validation Accuracy : 0.5424000024795532
Epoch 40  - Validation Accuracy : 0.603600025177002
Epoch 50  - Validation Accuracy : 0.6456000208854675
Epoch 60  - Validation Accuracy : 0.675599992275238
Epoch 70  - Validation Accuracy : 0.698199987411499
Epoch 80  - Validation Accuracy : 0.7174000144004822
Epoch 90  - Validation Accuracy : 0.7314000129699707
Trained model saved
Test Accuracy : 0.7418000102043152
"""