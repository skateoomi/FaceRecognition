def handleImageDK(training, validation, f, image, output, val_out):
    if f.count('val') == 1:
        validation.append(image)
        # validation.append(rotate_image(image))
        if f.count('donald') == 0:
            val_out.append([0, 1])
            # val_out.append([0, 1])
        elif f.count('kermit') == 0:
            val_out.append([1, 0])
            # val_out.append([1, 0])

    else:
        training.append(image)
        # training.append(rotate_image(image))
        if f.count('donald') == 0:
            """NO ES DONALD"""
            output.append([0, 1])
            # output.append([0, 1])
        elif f.count('kermit') == 0:
            """NO ES KERMIT"""
            output.append([1, 0])
            # output.append([1, 0])

def handleImageNumbers(training, f, image, output):
    training.append(image)
    if (f.count('zero') == 1):
        output.append([1, 0, 0, 0, 0])
    elif (f.count('one') == 1):
        output.append([0, 1, 0, 0, 0])
    elif (f.count('two') == 1):
        output.append([0, 0, 1, 0, 0])
    elif (f.count('three') == 1):
        output.append([0, 0, 0, 1, 0])
    elif (f.count('four') == 1):
        output.append([0, 0, 0, 0, 1])