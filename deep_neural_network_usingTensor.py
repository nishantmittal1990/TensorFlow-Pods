from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# we will use MNIST dataset provided by TensorFlow, which batches and one-hot encodes for you.
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

learning_rate = 0.001
training_epochs = 20
batch_size = 128
display_step = 1

n_input = 784
n_classes = 10

"""The focus here is to learn architecture of multilayer neural network, not parameter tunning"""

# hidden_layer parameter
n_hidden_layer = 256 # layer no. of features

#n_hidden_layer determines the size of hidden layer in the neural network. this is also known as width of a layer

# stored layer weights and bias

weights = {
    'hidden_layer':tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'out' : tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}

biases = {
    'hidden_layer':tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

"""
Deep neural network uses multiple layers in each layer require their own weights and bias.
The 'hidden_layer' weights and bias is for hidden layer. The 'out' weights and bias is for output layer.
if neural networks are deeper, there would be weights and bias for each additional layer
"""
# Mnist data is made of 28px by 28px images with a single channel

x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

# tf.reshape() function reshapes the 28px by 28px matrics in * into row vector of 784px.
x_flat = tf.reshape(x, [-1, n_input])

# hidden layer using relu function
layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)
logits = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])

# define loss and optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epochs in range(training_epochs):

        total_batches = int(mnist.train.num_examples/batch_size)
        # loop over batches
        for batch in range(total_batches):
            # MNIST library in tensorflow provides the ability to receive the dataset in batches.
            # calling the mnist.train.next_batch() function returns a subset of training data.
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})