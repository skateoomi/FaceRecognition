import tensorflow as tf
import tensorflow as np


class Convolution:
    def __init__(self, name, inputs, size_filter, num_filters, tftype, non_linear_f, strides=1, use_relu=True):
        shape = [size_filter[0], size_filter[1], int(inputs.get_shape()[-1]), num_filters]
        with tf.variable_scope(name):
            std_dev = np.sqrt(2.0 / inputs.get_shape()[1:4].num_elements())
            self.weight = tf.get_variable('weight', shape, tftype, tf.random_normal_initializer(stddev=std_dev), trainable=True)
            self.convolution = tf.nn.conv2d(inputs, self.weight, [1, strides, strides, 1], 'SAME')
            self.bias = tf.get_variable('bias', dtype=tftype, shape=num_filters, initializer=tf.zeros_initializer())
            self.convolution += self.bias

            if use_relu:
                mean, variance = tf.nn.moments(self.convolution, axes=[0, 1, 2], keep_dims=True)
                self.convolution = tf.nn.batch_normalization(self.convolution, mean, variance, None, None, 1e-10)
                self.convolution = non_linear_f(self.convolution)