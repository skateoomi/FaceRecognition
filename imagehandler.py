from cv2 import imread, imshow, waitKey, destroyAllWindows

def handleImage(training, validation, f) {
    image = imread(path + f)

    image = reshape(image, res)

    # imshow('wabalaba dubdub', image)
    # waitKey(0)
    # destroyAllWindows()

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
        images.append(image)
        # images.append(rotate_image(image))
        if f.count('donald') == 0:
            """NO ES DONALD"""
            output.append([0, 1])
            # output.append([0, 1])
        elif f.count('kermit') == 0:
            """NO ES KERMIT"""
            output.append([1, 0])
            # output.append([1, 0])
}