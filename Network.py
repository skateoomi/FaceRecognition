import tensorflow as tf


class Network:
    std_dev = 0.05
    lmda = 1e-6

    def __init__(self, name, listNames, previous, outputs, convolutions, fcnodes, reuse=None):
        self.non_linear_conv = tf.nn.relu
        self.non_linear_fc = tf.nn.tanh
        with tf.variable_scope(name, reuse=reuse):
            branch_prev = None
            filter_list = []
            res_prev = None
            convs = 0
            bottles = 0
            for name in listNames:
                if name.count('conv'):

                    conv = self._createConvolution(
                        name=name + str(convs),
                        inputs=previous,
                        size_filter=convolutions[name + str(convs)]['size_filter'],
                        num_filters=convolutions[name + str(convs)]['num_filters'],
                        use_relu=convolutions[name + str(convs)]['non_linear']
                    )
                    previous = conv['conv']
                    self.weight = conv['weight']
                    convs += 1

                if name.count('bottle'):
                    bottle = self._createBottleneck(
                        name + str(bottles),
                        previous
                    )
                    previous = bottle
                    bottles += 1

                if name.count('pool'):
                    pool = tf.nn.max_pool(
                        value=previous,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='VALID'
                    )
                    previous = pool

                if name.count('filter'):
                    fil_con = self._filter_concat(name, filter_list)
                    filter_list = []
                    previous = fil_con
                    branch_prev = None

                if name.count('branch'):
                    assert (branch_prev is None)
                    branch_prev = previous

                if name.count('tie'):
                    filter_list.append(previous)
                    previous = branch_prev

                if name.count('opres'):
                    assert (res_prev is None)
                    res_prev = previous

                if name.count('closeres'):
                    previous = self._createResConnection(name, residual=previous, original=res_prev)

            previous, num_fea = self._flatten_layer(previous)
            self.before_fc = previous

            for fc in fcnodes:
                previous = self._createfc('fc' + str(fc), previous, fc, non_linear_use=(fc == fcnodes[-1]))

            """LOSS FUNCTION"""
            self.output = previous
            self.softmax = tf.nn.softmax_cross_entropy_with_logits(logits=previous, labels=outputs)
            self.loss = tf.reduce_mean(self.softmax)
            self.optimizer = tf.train.AdamOptimizer(self.lmda).minimize(self.loss)
            self.off = tf.reduce_sum(previous, 0)

    def _createfc(self, name, inputs, num_outputs, non_linear_use=True):
        num_inputs = inputs.get_shape()[-1]
        with tf.variable_scope(name):
            weight = tf.get_variable('weight', [num_inputs, num_outputs], tf.float32,
                                     tf.random_normal_initializer(stddev=self.std_dev))
            bias = tf.get_variable('bias', [num_outputs], tf.float32, tf.ones_initializer())
            output = tf.matmul(inputs, weight)
            output += bias
            if non_linear_use:
                output = self.non_linear_fc(output)
        return output

    def _createConvolution(self, name, inputs, size_filter, num_filters, strides=1, use_relu=True):
        shape = [size_filter[0], size_filter[1], int(inputs.get_shape()[-1]), num_filters]
        with tf.variable_scope(name):
            weight = tf.get_variable('weight', shape, tf.float32, tf.random_normal_initializer(stddev=self.std_dev))
            convolution = tf.nn.conv2d(inputs, weight, [1, strides, strides, 1], 'SAME')
            bias = tf.get_variable('bias', dtype=tf.float32, shape=num_filters, initializer=tf.ones_initializer())
            convolution += bias

            if use_relu:

                convolution = self.non_linear_conv(convolution)

            return {'conv': convolution, 'weight': weight}

    def _createBottleneck(self, name, inputs, use_relu=True):
        shape = [1, 1, int(inputs.get_shape()[-1]), 1]
        with tf.variable_scope(name):
            weight = tf.get_variable('weight', shape, tf.float32, tf.random_normal_initializer(stddev=self.std_dev))
            convolution = tf.nn.conv2d(inputs, weight, [1, 1, 1, 1], 'SAME')
            bias = tf.get_variable('bias', dtype=tf.float32, shape=[1], initializer=tf.ones_initializer())
            convolution += bias
            if use_relu:
                convolution = self.non_linear_conv(convolution)
        return convolution

    def _flatten_layer(self, layer):
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

    def _filter_concat(self, name, filters):
        with tf.variable_scope(name):
            filterconcat = tf.zeros(filters[0].shape[1:])
            num_channels = filters[0].get_shape()[-1]
            i = 0
            for filter in filters:
                with tf.variable_scope('filter_weight' + str(i)):
                    weight = tf.get_variable('weight', [num_channels], tf.float32,
                                             tf.random_normal_initializer(stddev=self.std_dev))
                    aux = tf.multiply(weight, filter)
                    filterconcat = tf.add(aux, filterconcat)
                i += 1
            filterconcat = self.non_linear_conv(filterconcat)
        return filterconcat

    def _createResConnection(self, name, residual, original):
        with tf.variable_scope(name):
            weight = tf.get_variable('weight', original.get_shape()[1:], tf.float32,
                                     tf.random_normal_initializer(stddev=self.std_dev))
            res = tf.add(tf.multiply(original, weight), residual)
            res = tf.nn.relu(res)
            return res

    def depictlearnt(self, session, image, weight):
        sol = np.array(session.run(self.non_linear_conv(tf.nn.conv2d(image, weight, [1, 1, 1, 1], 'SAME'))))[0]
