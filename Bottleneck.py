import tensorflow as tf
import numpy as np


class Bottleneck:
    def __init__(self, name, inputs, tftype, non_linear_f, use_relu=True):
        shape = [1, 1, int(inputs.get_shape()[-1]), 1]
        with tf.variable_scope(name):
            std_dev = np.sqrt(2.0 / inputs.get_shape()[1:4].num_elements())
            self.weight = tf.get_variable('weight', shape, tftype, tf.random_normal_initializer(stddev=std_dev))
            self.convolution = tf.nn.conv2d(inputs, self.weight, [1, 1, 1, 1], 'SAME')
            self.bias = tf.get_variable('bias', dtype=tftype, shape=[1], initializer=tf.ones_initializer())
            self.convolution += self.bias
            if use_relu:
                self.convolution = non_linear_f(self.convolution)