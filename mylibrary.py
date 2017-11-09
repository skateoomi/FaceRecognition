import os
import numpy as np
from image_handler import *
from cv2 import imread, imshow, waitKey, destroyAllWindows, namedWindow, VideoCapture
import cPickle as pk

output_opts = ['soria', 'rob', 'omar', 'pouli', 'borja', '']


def loadImages(res, path):
    print('Loading images...')
    files_train = os.listdir(path + 'train/')
    files_val = os.listdir(path + 'val/')
    images = []
    output = []
    validation = []
    val_out = []
    if path.count('images') == 1:
        size_out = 2
        size_out = 6
        picklename = './loaded_images/DK.p'
    else:
        size_out = 10
        picklename = './loaded_images/numbers.p'
    try:
        aux = pk.load(open(picklename, 'rb'))
        print('Using pickled array')
        return aux['training_input'], aux['training_output'], aux['validation_input'], \
               aux['validation_output'], size_out
    except:
        print('Non_exiting pickle, processing images...')
        for f in files_train:
            image = imread(path + 'train/' + f)
            image = reshape(image, res)
            handle_set(output_opts, f, image, images, output)

        for f in files_val:
            image = imread(path + 'val/' + f)
            image = reshape(image, res)
            handle_set(output_opts, f, image, validation, val_out)

        images = np.array(images, dtype=np.float64)
        validation = np.array(validation, dtype=np.float64)
        images -= np.mean(images, axis=0)
        validation -= np.mean(validation, axis=0)
        images /= (np.std(images, axis=0) + 1e-11)
        validation /= (np.std(validation, axis=0) + 1e-11)
        pack = {'training_input': images, 'training_output': output, 'validation_input': validation,
                'validation_output': val_out}
        pk.dump(pack, open(picklename, 'wb'))
        return images, np.array(output), validation, np.array(val_out), size_out


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
        print("Height: ", len(new_image[0]))
        print("Width: ", len(new_image))
        print("Channels: ", len(new_image[0][0]))

    return np.array(new_image)