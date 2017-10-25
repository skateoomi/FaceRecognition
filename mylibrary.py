import tensorflow as tf
import os
import numpy as np
from imagehandler import handleImageDK, handleImageNumbers
from cv2 import imread, imshow, waitKey, destroyAllWindows


def loadImages(res, path):
    files = os.listdir(path)
    images = []
    output = []
    validation = []
    val_out = []

    for f in files:
        if (f.count('zero') == 0):
            continue
        image = imread(path + f)

        image = reshape(image, res)

        imshow('wabalaba dubdub', image)
        waitKey(0)
        destroyAllWindows()
        # handleImageDK(images, validation, f, image, output, val_out)
        handleImageNumbers(images, f, image, output)

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
        print("Height: ", len(new_image[0]))
        print("Width: ", len(new_image))
        print("Channels: ", len(new_image[0][0]))

    return np.array(new_image)


def rotate_image(image):
    res = len(image[0])
    rotated = np.zeros((res, res, 3))
    for i in range(res):
        for j in range(res):
            rotated[i][j] = image[i][res - j - 1]

    return rotated


# def createDeconv():
#     tf.nn.conv2d_transpose()

