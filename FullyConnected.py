import tensorflow as tf
import numpy as np

class FullyConnected:
    def __init__(self, name, inputs, num_outputs, tftype, non_linear_f=None, non_linear_use=True):
        num_inputs = inputs.get_shape()[-1]
        with tf.variable_scope(name):
            std_dev = np.sqrt(2.0 / inputs.get_shape()[1:-1].num_elements())

            self.weight = tf.get_variable('weight', [num_inputs, num_outputs], tftype,
                                     tf.random_normal_initializer(std_dev), trainable=True)
            self.bias = tf.get_variable('bias', [num_outputs], tftype, tf.zeros_initializer())
            self.output = tf.matmul(inputs, self.weight)
            self.output += self.bias
            if non_linear_use:
                mean, variance = tf.nn.moments(self.output, axes=[0, 1])
                self.output = tf.nn.batch_normalization(self.output, mean, variance, None, None, 1e-8)
                self.output = non_linear_f(self.output)

