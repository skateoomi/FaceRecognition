from mylibrary import *
from Network import *
import time
import matplotlib.pyplot as plt

path = "./numbers/"
res = 150
training_set_division = 10
num_networks = 1
actual_num_networks = num_networks

structure = ['conv', 'bottle', 'conv', 'bottle', 'pool', 'conv', 'bottle', 'pool', 'conv', 'bottle', 'conv', 'pool']
num_filters = [8, 8, 8, 32, 32]
size_filters = [[11, 1], [1, 11], [7, 7], [5, 1], [1, 5]]
strides = []
non_linearities = [1, 1, 1, 1, 1]
fully_connected = []


def start_train():
    global actual_num_networks
    accuracy_track_train = []
    accuracy_track_val = []
    loss_track_train = []
    loss_track_val = []
    for epoch in range(2000):
        print("Epoch: ", epoch)
        mean_acc = 0
        mean_loss = 0
        for net in range(len(network)):
            act_network = network[net]
            for i in range(training_set_division):
                """TRAIN"""
                actual_inputs = image[i * int(num_images / training_set_division): (i + 1) * int(
                    num_images / training_set_division)]
                actual_outputs = output[i * int(num_images / training_set_division): (i + 1) * int(
                    num_images / training_set_division)]

                data_interested_in = {
                    'loss': act_network.loss,
                    'optimizer': act_network.optimizer,
                    'accuracy': act_network.accuracy,
                    'off': act_network.off,
                    'before fc': act_network.before_fc,
                    'output': act_network.output,
                    'softmax': act_network.softmax,
                    'gradients to before fc': act_network.gradients_to_fc
                }

                train = act_network(data_interested_in, input=actual_inputs, output=actual_outputs)

                if (epoch + 1) % 250 == 0:
                    act_network.deconv_net(0, res)

                mean_acc += train['accuracy'] / training_set_division
                mean_loss += train['loss'] / training_set_division

            """VALIDATE"""
            data_interested_in = {
                'accuracy': act_network.accuracy,
                'loss': act_network.loss
            }

            val = act_network(data_interested_in, input=valimage, output=valoutput)

        accuracy_track_train.append(mean_acc)
        accuracy_track_val.append(val['accuracy'] * 100)
        loss_track_train.append(mean_loss)
        loss_track_val.append(val['loss'])
        plt.subplot(2, 2, 1)
        plt.cla()
        plt.plot(accuracy_track_train)
        plt.subplot(2, 2, 2)
        plt.cla()
        plt.plot(accuracy_track_val)
        plt.subplot(2, 2, 3)
        plt.cla()
        plt.plot(loss_track_train)
        plt.subplot(2, 2, 4)
        plt.cla()
        plt.plot(loss_track_val)
        plt.draw()
        plt.pause(0.000000000000001)
        # print("Errors: %.4f" % mean_loss)
        print("Errors: %.4f %.4f" % (mean_loss, val['loss']))
        # print("Accuracy(%%): %.1f" % (mean_acc * 100))
        print("Accuracy(%%): %.1f %.1f" % (mean_acc * 100, val['accuracy'] * 100))
        print("------------------------------------------------------------")


    return True


if __name__ == "__main__":
    t_before = time.time()
    """LOAD THE DATA"""
    image, output, valimage, valoutput, size_output = loadImages(res, path)
    print('Load took ' + str(time.time()-t_before) + 's')
    num_images = len(image)
    print("TRAINING NETWORK WITH ", num_images, " TRAINING IMAGES AND ", len(valimage), " VALIDATION IMAGES.")
    print ("___________________________________________________________")

    """CREATE NETWORK"""
    convolutions = {}
    for i in range(len(num_filters)):
        aux = {'size_filter': size_filters[i],
               'num_filters': num_filters[i],
               'non_linear': non_linearities[i]

        }
        convolutions['conv' + str(i)] = aux

    network = [0] * num_networks

    for network_num in range(num_networks):
        network[network_num] = Network('network' + str(network_num), size_output, res, structure, convolutions, fully_connected)

    while start_train() is None:
        print('RESTARTING...')
        print('_______________________________________________________')
