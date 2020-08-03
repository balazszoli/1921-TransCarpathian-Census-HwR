import os
import cv2
import matplotlib.pyplot as plt
# import keras
# import tensorflow as tf

impath='images-cropped1/111x77/'

# Saransh Gupta, https://www.quora.com/How-can-I-read-multiple-images-in-Python-presented-in-a-folder
def readFilesIn(path):
    return [d for d in os.listdir(path) if not os.path.isdir(os.path.join(path, d))]


filelist = readFilesIn(impath)
images = []
labels = []

img = cv2.imread(impath+filelist[0], cv2.IMREAD_GRAYSCALE)
width, height = img.shape[1], img.shape[0]
crop_out = int((width-height)/2)

for f in filelist:
    img = cv2.imread(impath+f, cv2.IMREAD_GRAYSCALE)
    # crop
    if crop_out > 0:
        cropped = img[0:height, crop_out:width-crop_out]
    else:
        cropped = img[-crop_out:height+crop_out, 0:width]
    #resize
    resized = cv2.resize(cropped,(28,28),interpolation = cv2.INTER_AREA)

    # plt.imshow(img, cmap='gray', interpolation='bicubic')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()
    cv2.imwrite(impath+'resized/' + f, resized)
    # y = input('label= ')
    #
    # images.append(img)
    # labels.append(y)

# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# plt.imshow(x_train[0], cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()
# cv2.imwrite(impath+'mnist-x0.jpg', x_train[0])
