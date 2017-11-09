import cv2
import os
import imutils
import math
import numpy as np

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.004*peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            if not (ar >= 0.95 and ar <= 1.05):
                return approx
        return None

files = os.listdir('../images/')
for f in files:
    image = cv2.imread('../images/' + f)
    resized = imutils.resize(image, width=600)
    ratio = image.shape[0] / float(resized.shape[0])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    sd = ShapeDetector()
    file_angle = None
    width_positions = []
    height_positions = []
    # cv2.imshow("Image", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for c in cnts:
        M = cv2.moments(c)
        cX = int((M["m10"] / (M["m00"] + 1e-10)))
        cY = int((M["m01"] / (M["m00"] + 1e-10)))
        points = sd.detect(c)
        if points is None or points[0][0][1] < 400:
            continue

        """Calculate de angle of the file just the first time"""
        if file_angle is None:
            diff_w = abs(points[0][0][0] - points[1][0][0])
            diff_h = abs(points[0][0][1] - points[1][0][1])
            hip = math.sqrt(math.pow(diff_w, 2) + math.pow(diff_h, 2))
            if diff_w < 10:
                file_angle = math.asin(diff_w / hip)
            if diff_h < 10:
                file_angle = math.acos(diff_w / hip)
            if file_angle is None:
                print "Error: Angle was not calculated!"
                exit(0)
            print file_angle

        """Calculate min and max positions of the rectangle"""
        min_w = max_w = points[3][0][0]
        min_h = max_h = points[3][0][1]
        for i in range(3):
            p = points[i][0]
            if p[0] < min_w:
                min_w = p[0]
            if p[0] > max_w:
                max_w = p[0]
            if p[1] < min_h:
                min_h = p[1]
            if p[1] > max_h:
                max_h = p[1]

        """If is too big or too small continue"""
        width = max_w - min_w
        if width < 300 or width > 1000:
            continue

        """Calculate the min and max positions of width to calculate widths of file"""
        exist = False
        for value in width_positions:
            diff = abs(max_w - value)
            if diff < 10:
                exist = True
        if not exist:
            width_positions.append(max_w)
        exist = False
        for value in width_positions:
            diff = abs(min_w - value)
            if diff < 10:
                exist = True
        if not exist:
            width_positions.append(min_w)

        """Calculate the min and max positions of height to calculate heights of file"""
        exist = False
        for value in height_positions:
            diff = abs(max_h - value)
            if diff < 10:
                exist = True
        if not exist:
            height_positions.append(max_h)
        exist = False
        for value in height_positions:
            diff = abs(min_h - value)
            if diff < 10:
                exist = True
        if not exist:
            height_positions.append(min_h)

        """DRAW"""
        # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        # # cv2.putText(resized, 'o', (points[0][0][0], points[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
        # aux = imutils.resize(image, width=500)
        # cv2.imshow("Image", aux)
        # cv2.waitKey(10000)
        # cv2.destroyAllWindows()

    """I AM ONLY DEALING WITH PRICK RIGHT NOW1!!!"""
    if len(width_positions) == 2:
        continue

    """DOWNSCALE"""
    down_width = 500
    aux = imutils.resize(image, width=down_width)
    ratio = float(aux.shape[0]) / image.shape[0]
    width_positions = list(map(lambda x: int(x * ratio), width_positions))
    height_positions = list(map(lambda x: int(x * ratio), height_positions))

    """Try to solve posible errors in recognition"""
    f_width_positions = width_positions[:]
    for i in width_positions:
        if i < 150:
            f_width_positions.remove(i)
    width_positions = f_width_positions
    """Sort"""
    width_positions = sorted(width_positions)
    height_positions = sorted(height_positions)
    print width_positions
    print height_positions

    """If last width is too little, add the value which I empirically checked is gonna be the last width value"""
    if width_positions[-1] < 450:
        width_positions.append(483)


    """In the second line starts the relevant stuff"""
    file_width = width_positions[-1] - width_positions[0]
    file_height = height_positions[-1] - height_positions[0]
    print ratio


    def new_image(image, p, width, height):
        new_image = []
        for h in range(height):
            new_row = []
            for w in range(width):
                new_row.append(image[h + p[1]][w + p[0]])
            new_image.append(new_row)
        return np.array(new_image)


    # show the output image
    cv2.imshow("Image", new_image(aux, [width_positions[0], height_positions[0]], file_width, file_height))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
