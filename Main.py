from mylibrary import *
from random import random
from math import floor

size_output = 2
res = 50
training_set_division = 4

if __name__ == "__main__":
    session = tf.Session()

    """PLACE HOLDER INIT"""
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, res, res, 3), name='inputs')
    outputs = tf.placeholder(dtype=tf.float32, shape=(None, size_output), name='outputs')
    isTrain = tf.placeholder(dtype=tf.bool, shape=(), name='isTrain')
    ytrue = tf.argmax(outputs, dimension=1)

    """LOAD THE DATA"""
    image, output, valimage, valoutput = loadImages(res)
    num_images = len(image)

    """CREATE NETWORK"""
    num_filters = [50, 50, 100, 100]
    size_filters = [11, 11, 5, 5]
    convolutions = {}
    for i in range(len(num_filters)):
        aux = {'size_filter': size_filters[i], 'num_filters': num_filters[i]}
        convolutions['conv' + str(i + 1)] = aux

    structure = ['branch', 'conv1', 'bootle1', 'tie', 'conv2', 'bootle2', 'tie', 'filter',
                 'pool1', 'conv3', 'bootle3', 'conv4', 'bottle4', 'pool2']
    network = createNetwork(structure, inputs, convolutions, [2048, size_output])

    """LOSS FUNCTION"""
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=outputs)
    loss = tf.reduce_mean(softmax)

    """ACCURACY"""
    ypred = tf.argmax(tf.nn.softmax(network), dimension=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(ypred, ytrue), tf.float32))

    optimizer = tf.train.AdamOptimizer(1e-7).minimize(loss)

    session.run(tf.global_variables_initializer())

    for epoch in range(2000):
        value = int(floor(random() * num_images))
        mean_acc = 0
        mean_loss = 0
        for i in range(training_set_division):
            """TRAIN"""
            train = session.run([loss, optimizer, accuracy], feed_dict={
                inputs: image[i * (num_images / training_set_division): (i + 1) * (num_images / training_set_division)],
                outputs: output[i * (num_images / training_set_division): (i + 1) * (num_images / training_set_division)],
                isTrain: True
            })
            mean_acc += train[2] / training_set_division
            mean_loss += train[0] / training_set_division

        """VALIDATE"""
        val = session.run([accuracy, loss, ytrue, ypred], feed_dict={
            inputs: valimage,
            outputs: valoutput,
            isTrain: False
        })

        print "Errors: ", mean_loss, val[1]
        # print "FC2: ", train[2][0]
        print "Accuracy: ", mean_acc * 100, "%", val[0] * 100, "%"
        # print "Difference: ", val[2], " ", val[3]



