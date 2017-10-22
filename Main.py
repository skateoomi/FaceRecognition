from mylibrary import *
from random import random
from math import floor

size_output = 2
res = 50
training_set_division = 1
num_networks = 1
actual_num_networks = num_networks

def start_train():
    global actual_num_networks

    for epoch in range(2000):
        value = int(floor(random() * num_images))
        print("Epoch: ", epoch)
        mean_acc = 0
        mean_loss = 0
        for net in range(len(network)):

            act_network = network[net]
            # print act_network['name']
            for i in range(training_set_division):
                """TRAIN"""
                actual_inputs = image[i * int(num_images / training_set_division): (i + 1) * int(num_images / training_set_division)]
                actual_outputs = output[i * int(num_images / training_set_division): (i + 1) * int(num_images / training_set_division)]

                train = session.run([act_network['loss'], act_network['optimizer'],
                                     accuracy, act_network['off']], feed_dict={
                    inputs: actual_inputs,
                    outputs: actual_outputs,
                    isTrain: True
                })
                # print(train[4])
                if train[3][0] == 0 or train[3][1] == 0:
                    print("RESTART THE DEAD NETWORK: ", net)
                    network[net] = createNetwork('network' + str(actual_num_networks), structure, inputs, outputs, convolutions, [2048, size_output])
                    actual_num_networks += 1
                    session.run(tf.global_variables_initializer())
                    return None

                mean_acc += train[2] / training_set_division
                mean_loss += train[0] / training_set_division

        """VALIDATE"""
        # val = session.run([accuracy, network[0]['loss'], ytrue, ypred], feed_dict={
        #     inputs: valimage,
        #     outputs: valoutput,
        #     isTrain: False
        # })

        print("Errors: %.4f" % mean_loss)
        # print("Errors: %.4f %.4f" % (mean_loss, val[1]))
        # print "FC2: ", train[2][0]
        print("Accuracy(%%): %.1f" % (mean_acc * 100))
        # print("Accuracy(%%): %.1f %.1f" % (mean_acc * 100, val[0] * 100))
        # print "Difference: ", val[2], " ", val[3]
        print("------------------------------------------------------------")


if __name__ == "__main__":
    session = tf.Session()

    """PLACE HOLDER INIT"""
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, res, res, 3), name='inputs')
    outputs = tf.placeholder(dtype=tf.float32, shape=(None, size_output), name='outputs')
    isTrain = tf.placeholder(dtype=tf.bool, shape=(), name='isTrain')
    ytrue = tf.argmax(outputs, dimension=1)

    """LOAD THE DATA"""
    image, output, valimage, valoutput = loadImages(res, "./images/")
    num_images = len(image)
    print("TRAINING NETWORK WITH ", num_images, " TRAINING IMAGES AND ", len(valimage), " VALIDATION IMAGES.")
    print ("___________________________________________________________")
    """CREATE NETWORK"""
    num_filters = [64, 64,
                   128, 128, 128,
                   256, 256, 512]
    size_filters = [[11, 11], [7, 7],
                    [5, 5], [3, 1], [1, 3],
                    [3, 3], [3, 3], [3, 3]]
    non_linearities = [1, 1, 0,
                       1, 1, 1,
                       1, 1, 1]
    convolutions = {}
    for i in range(len(num_filters)):
        aux = {'size_filter': size_filters[i], 'num_filters': num_filters[i], 'non_linear': non_linearities[i]}
        convolutions['conv' + str(i)] = aux

    structure = ['conv', 'opres', 'conv', 'closeres',
                 'branch', 'conv', 'bottle', 'tie', 'conv', 'conv', 'bottle', 'tie', 'filter',
                 'pool1',
                 'branch', 'conv', 'bottle', 'tie', 'conv', 'bottle', 'tie', 'conv', 'bottle', 'tie', 'filter2']

    network = [0] * num_networks
    outs = []
    for network_num in range(num_networks):
        network[network_num] = createNetwork('network' + str(network_num), structure, inputs, outputs, convolutions, [2048, size_output])
        outs.append(network[network_num]['output'])

    """ACCURACY"""
    intermediate = tf.reduce_mean(outs, 0)
    ypred = tf.argmax(tf.nn.softmax(intermediate), dimension=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(ypred, ytrue), tf.float32))

    session.run(tf.global_variables_initializer())

    while start_train() is None:
        print('RESTARTING...')
        print('_______________________________________________________')

