import numpy as np


def handleImageNumbers(training, f, image, output):
    training.append(image)
    if f.count('zero') == 1:
        output.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif f.count('one') == 1:
        output.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif f.count('two') == 1:
        output.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif f.count('three') == 1:
        output.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif f.count('four') == 1:
        output.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif f.count('five') == 1:
        output.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif f.count('six') == 1:
        output.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif f.count('seven') == 1:
        output.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif f.count('eight') == 1:
        output.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    elif f.count('nine') == 1:
        output.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


def handle_set(tags, f, image, input, output):
    input.append(image)
    input.append(rotate_image(image))
    dif_opts = len(tags)
    counter = 0
    for name in tags:
        if f.count(name):
            aux = [0] * dif_opts
            aux[counter] = 1
            output.append(aux)
            output.append(aux)
            break
        counter += 1


def rotate_image(image):
    res = len(image[0])
    rotated = np.zeros((res, res, 3))
    for i in range(res):
        for j in range(res):
            rotated[i][j] = image[i][res - j - 1]

    return rotated


