import tensorflow as tf
from cv2 import imread, imshow, waitKey, destroyAllWindows
import os
import numpy as np
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops


non_linear = tf.nn.relu

def createWeight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))


def createBias(shape):
    return tf.Variable(tf.constant(0.05, shape=shape))


def createfc(name, inputs, num_outputs):
    num_inputs = inputs.get_shape()[-1]
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [num_inputs, num_outputs], tf.float32, tf.random_normal_initializer())
        bias = tf.get_variable('bias', [num_outputs], tf.float32, tf.ones_initializer())
        output = tf.matmul(inputs, weight)
        output += bias
        output = non_linear(output)
    return output


def createConvolution(name, inputs, size_filter, num_filters, strides=1, use_relu=True, isTrain=True):
    shape = [size_filter, size_filter, int(inputs.get_shape()[-1]), num_filters]
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape, tf.float32, tf.random_normal_initializer(0.05))
        # weight_norm = tf.nn.l2_normalize(weight.initialized_value(), [0, 1, 2])
        convolution = tf.nn.conv2d(inputs, weight, [1, strides, strides, 1], 'SAME')
        bias = tf.get_variable('bias', dtype=tf.float32, shape=num_filters, initializer=tf.ones_initializer())
        convolution += bias
        if use_relu:
            convolution = non_linear(convolution)
        # gamma = tf.get_variable('gamma', dtype=tf.float32, shape=sop, initializer=tf.ones_initializer)
        # beta = tf.get_variable('beta', dtype=tf.float32, shape=sop, initializer=tf.zeros_initializer)
        # moving_mean = tf.get_variable('moving_mean', dtype=tf.float32, shape=sop, initializer=tf.zeros_initializer, trainable=False)
        # moving_variance = tf.get_variable('moving_variance', dtype=tf.float32, shape=sop, initializer=tf.ones_initializer, trainable=False)
        # mean, variance = tf.nn.moments(convolution, [0, 1, 2])
        # update_moving_mean = moving_averages.assign_moving_average(moving_mean,
        #                                                            mean, .9997)
        # update_moving_variance = moving_averages.assign_moving_average(
        #     moving_variance, variance, .9997)
        # tf.add_to_collection('update_ops', update_moving_mean)
        # tf.add_to_collection('update_ops', update_moving_variance)
        # mean, variance = control_flow_ops.cond(
        #     isTrain, lambda: (mean, variance),
        #     lambda: (moving_mean, moving_variance))
        #
        # convolution = tf.nn.batch_normalization(convolution, mean, variance, beta, gamma, 0.001)

    return convolution


def createBottleneck(name, inputs, use_relu=True):
    shape = [1, 1, int(inputs.get_shape()[-1]), 1]
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape, tf.float32, tf.random_normal_initializer(0.05))
        # weight_norm = tf.nn.l2_normalize(weight.initialized_value(), [0, 1, 2])
        convolution = tf.nn.conv2d(inputs, weight, [1, 1, 1, 1], 'SAME')
        # bias = createBias([1])
        # convolution += bias
        if use_relu:
            convolution = non_linear(convolution)
    return convolution


def loadImages(res):
    files = os.listdir("images/")
    images = []
    output = []
    validation = []
    val_out = []
    for f in files:
        image = imread("images/" + f)

        image = reshape(image, res)

        imshow('wabalaba dubdub', rotate_image(image))
        waitKey(0)
        destroyAllWindows()

        if f.count('val') == 1:
            validation.append(image)
            if f.count('donald') == 0:
                val_out.append([0, 1])
            elif f.count('kermit') == 0:
                val_out.append([1, 0])
        else:
            images.append(image)
            if f.count('donald') == 0:
                """NO ES DONALD"""
                output.append([0, 1])
            elif f.count('kermit') == 0:
                """NO ES KERMIT"""
                output.append([1, 0])

    return np.array(images), np.array(output), np.array(validation), np.array(val_out)




def reshape(image, res, verbose=False):
    height = len(image)
    width = len(image[0])
    width_ratio = res / float(width - 1)
    height_ratio = res / float(height - 1)
    new_image = []
    counter_height = 0
    for i in range(height):
        counter_height += height_ratio
        if counter_height < 1:
            continue
        counter_height -= 1
        new_row = []
        counter_width = 0
        for j in range(width):
            counter_width += width_ratio
            if counter_width < 1:
                continue
            counter_width -= 1
            add_value = image[i][j] / float(255)
            new_row.append(add_value)
        new_image.append(new_row)


    if verbose:
        print "Height: ", len(new_image[0])
        print "Width: ", len(new_image)
        print "Channels: ", len(new_image[0][0])

    return np.array(new_image)


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def filter_concat(name, filters):
    with tf.variable_scope(name):
        filterconcat = tf.zeros(filters[0].shape[1:])
        i = 0
        for filter in filters:
            with tf.variable_scope('filter_weight' + str(i)):
                weight = tf.get_variable('weight', [1], tf.float32, tf.random_normal_initializer(0.05))
                aux = tf.multiply(weight, filter)
                filterconcat = tf.add(aux, filterconcat)
            i += 1
        filterconcat = non_linear(filterconcat)
    return filterconcat


def createNetwork(listNames, previous, convolutions, fcnodes):
    branch_prev = None
    filter_list = []
    for name in listNames:
        if name.count('conv'):
            conv = createConvolution(
                name,
                previous,
                convolutions[name]['size_filter'],
                convolutions[name]['num_filters']
            )
            previous = conv

        if name.count('bottle'):
            bottle = createBottleneck(
                name,
                previous
            )
            previous = bottle

        if name.count('pool'):
            pool = tf.nn.max_pool(
                value=previous,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID'
            )
            previous = pool

        if name.count('filter'):
            fil_con = filter_concat(name, filter_list)
            previous = fil_con
            branch_prev = None

        if name.count('branch'):
            assert (branch_prev is None)
            branch_prev = previous

        if name.count('tie'):
            filter_list.append(previous)
            previous = branch_prev

    previous, num_fea = flatten_layer(previous)

    for fc in fcnodes:
        previous = createfc('fc' + str(fc), previous, fc)

    return previous


def rotate_image(image):
    res = len(image)
    rotated = np.array()
    for i in range(res):
        for j in range(res):
            rotated[i][j] = image[i][res - j - 1]
            print res - j - 1

    return rotated

