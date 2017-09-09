"""
Linear function in Tensor Flow
y=xW+b

Weights and bias in Tensor Flow
The Goal of training the neural network is to modify weights and bias to best predict the labels.
In order to use weights and bias, we need tensor that can be modified. We can't use tf.placeholder() and tf.constant(), since those tensor can't be modified.
This is where tf.Variable comes in.

The tf.Variable class creates a tensor with an initial value that can be modified, much like a normal Python variable. 
This tensor stores its state in the session, so you must initialize the state of the tensor manually. 
You'll use the tf.global_variables_initializer() function to initialize the state of all the Variable tensors.

Initialization
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
Initializing the weights with random numbers from a normal distribution is good practice. 
Randomizing the weights helps the model from becoming stuck in the same place every time you train it. 
You'll learn more about this in the next lesson, when you study gradient descent.

Similarly, choosing weights from a normal distribution prevents any one weight from overwhelming other weights. 
You'll use the tf.truncated_normal() function to generate random numbers from a normal distribution.

n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

Since the weights are already helping prevent the model from getting stuck, you don't need to randomize the bias. Let's use the simplest solution, setting the bias to 0.

tf.zeros()
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels)) --> The tf.zeros() function returns a tensor with all zeros.

"""
"""Quiz using linear function"""
## Aim in this quiz is to initialize the weights and bias
import tensorflow as tf

def get_weights(n_features, n_labels):
    """
    Return TensorFlow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow weights
    """
    # TODO: Return weights
    # as we are aware that weights keep on changing while training the neural network. We can't use tf.placeholder() and tf.constant(). instead, we will use tf.Variable()
    return tf.Variable(tf.truncated_normal((n_features, n_labels))) # Here apart from usign ts.variable we have also used truncated_normal.
    # truncated_normal generates random number from a normal distribution



def get_biases(n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """
    # TODO: Return biases
    # similarly for bias as well we will use tf.Variable and tf.Zeros - because we don't need to assign any random value to bias
    return tf.Variable(tf.zeros(n_labels))
    pass


def linear(input, w, b):
    """
    Return linear function in TensorFlow
    :param input: TensorFlow input
    :param w: TensorFlow weights
    :param b: TensorFlow biases
    :return: TensorFlow linear function
    """
    ## For linear we will use matmal function instead of multiplication because in case of matmul x*W is not same as W*x
    # TODO: Linear Function (xW + b)
    return tf.add(tf.matmul(input, w), b)