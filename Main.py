import cv2

if __name__ == "__main__":
    # inputs = tf.placeholder(dtype=i)
    image = cv2.imread("images/kermit3.jpg")
    print(image)