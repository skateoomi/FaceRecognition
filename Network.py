from cv2 import imshow, waitKey, destroyAllWindows
from Convolution import *
from FullyConnected import *
from Bottleneck import *

non_linear_conv = tf.nn.relu
non_linear_fc = tf.nn.relu
tftype = tf.float32


class Network:
    structure = []
    lmda = 1e-5

    def __init__(self, name, size_out, res, listNames, convolutions, fcnodes, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            self.session = tf.Session()
            self.size_output = size_out
            """PLACE HOLDER INIT"""
            self.inputs = tf.placeholder(dtype=tf.float32, shape=(None, res, res, 3), name='inputs')
            self.outputs = tf.placeholder(dtype=tf.float32, shape=(None, self.size_output), name='outputs')
            previous = self.inputs
            """CREATE"""
            branch_prev = None
            filter_list = []
            res_prev = None
            convs = 0
            bottles = 0
            for name in listNames:
                if name.count('conv'):
                    conv = Convolution(
                        name=name + str(convs),
                        inputs=previous,
                        size_filter=convolutions[name + str(convs)]['size_filter'],
                        num_filters=convolutions[name + str(convs)]['num_filters'],
                        use_relu=convolutions[name + str(convs)]['non_linear'],
                        non_linear_f=non_linear_conv,
                        tftype=tftype
                    )
                    previous = conv.convolution
                    convs += 1

                if name.count('bottle'):
                    previous = Bottleneck(
                        name + str(bottles),
                        previous,
                        tftype=tftype,
                        non_linear_f=non_linear_conv
                    ).convolution
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
                previous = FullyConnected('fc' + str(fc), previous, fc, tftype=tftype,
                                          non_linear_f=non_linear_fc, non_linear_use=True).output

            previous = FullyConnected('fc' + str(self.size_output), previous, self.size_output,
                                      tftype=tftype, non_linear_f=non_linear_fc, non_linear_use=True).output


            """LOSS FUNCTION"""
            self.output = previous
            self.softmax = tf.nn.softmax_cross_entropy_with_logits(logits=previous, labels=self.outputs)
            self.loss = tf.reduce_mean(self.softmax)
            self.optimizer = tf.train.AdamOptimizer(self.lmda).minimize(self.loss)
            self.off = tf.reduce_sum(previous, 0)
            self.gradients_to_fc = tf.gradients(self.loss, self.before_fc)

            """ACCURACY"""
            ytrue = tf.argmax(self.outputs, axis=1)
            intermediate = tf.reduce_mean([self.output], 0)
            ypred = tf.argmax(tf.nn.softmax(intermediate), axis=1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(ypred, ytrue), tftype))

            self.session.run(tf.global_variables_initializer())

    def __call__(self, *args, **kwargs):
        return self.session.run(args[0], feed_dict={
            self.inputs: kwargs['input'],
            self.outputs: kwargs['output']
        })

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
        global non_linear_conv
        with tf.variable_scope(name):
            filterconcat = tf.zeros(filters[0].shape[1:])
            num_channels = filters[0].get_shape()[-1]
            i = 0
            for filter in filters:
                with tf.variable_scope('filter_weight' + str(i)):
                    weight = tf.get_variable('weight', [num_channels], tftype,
                                             tf.random_normal_initializer(stddev=self.std_dev))
                    aux = tf.multiply(weight, filter)
                    filterconcat = tf.add(aux, filterconcat)
                i += 1
            filterconcat = non_linear_conv(filterconcat)
        return filterconcat

    def _createResConnection(self, name, residual, original):
        with tf.variable_scope(name):
            weight = tf.get_variable('weight', original.get_shape()[1:], tftype,
                                     tf.random_normal_initializer(stddev=self.std_dev))
            res = tf.add(tf.multiply(original, weight), residual)
            res = tf.nn.relu(res)
            return res

    def depictlearnt(self, image, weight):
        return np.array(self.session.run(self.non_linear_conv(tf.nn.conv2d(image, weight, [1, 1, 1, 1], 'SAME'))))[0]

    def deconv_net(self, target, res):
        randomImage = np.random.rand(1, res, res, 3)
        target = [self.session.run(tf.one_hot(target, self.size_output))]
        for i in range(2):
            grad, loss = self([tf.gradients(self.loss, self.inputs), self.loss], input=randomImage, output=target)
            randomImage = randomImage - grad[0]
            print loss
        imshow('New image', randomImage[0])
        waitKey(0)
        destroyAllWindows()
