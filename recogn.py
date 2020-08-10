import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
from imutils import rotate  # image rotation
from statistics import mode  # counting line angle mode, the most common value of angles

# filename = 'images/kajdano.png'  # moved images to images folder
# filename = 'images/kajdano2.png'
# filename = 'images/sebesh.png'
filename = 'images/table_tamplates/surname.png'

# TODO maybe we need filter out red color before grayscale???
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

from collections import OrderedDict
from collections import defaultdict

# Run edge detection
# TODO to achive better result we can adjust Canny parameters
canny_params = [50, 300]
dst = cv2.Canny(img, canny_params[0], canny_params[1], None, 3)  # 200 -> 300

# Find lines with Probabilistic Hough Line Transform (https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html)
lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 220, 10)

# Check if lines are parallel to layout edges, if not - rotate the image, and rerun Canny and HoughLinesP
a = []
for i in range(0, len(lines)):
    l = lines[i][0]
    angle = np.arctan2(l[3] - l[1], l[2] - l[0]) * 180. / np.pi
    a.append(angle)
angle_most = mode(a)
if abs(angle_most + 90) <= 1 or abs(angle_most) <= 1:  #
    angle = angle_most
else:
    if -45 < angle_most < 45:
        angle = angle_most
    elif angle_most <= -45:
        angle = angle_most + 90
    else:
        angle = angle_most - 90
    if abs(angle) > 0.7:
        img = rotate(img, angle)
        dst = cv2.Canny(img, canny_params[0], canny_params[1], None, 3)
        lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 220, 10)

# Copy edges to the images that will display the results in BGR
cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

horizontal = dict()
vertical = dict()
for i in range(0, len(lines)):
    l = lines[i][0]
    angel = np.arctan2(l[3] - l[1], l[2] - l[0]) * 180. / np.pi
    if (angel > -91 and angel < -89):
        vertical[l[0]] = [l[0], 0, l[2], cdst.shape[0]]
    if (angel == 0):
        horizontal[l[1]] = [0, l[1], cdst.shape[1], l[3]]

horizontal = OrderedDict(sorted(horizontal.items()))
vertical = OrderedDict(sorted(vertical.items()))


def reduc_lines(lines, threshould=10):
    buckets = []
    prev = next(iter(lines.keys()))
    current_bucket = []

    for key in lines.keys():
        if key > threshould + prev:
            if len(current_bucket) == 0:
                buckets.append([prev])
            buckets.append(current_bucket)
            current_bucket = []

        current_bucket.append(key)
        prev = key

    if len(current_bucket) > 0:
        buckets.append(current_bucket)

    res = []

    for bucket in buckets:
        mean = np.mean(bucket)
        res.append(lines[min(bucket, key=lambda x: abs(x - mean))])
    return res


horizontal = reduc_lines(horizontal, 10)
vertical = reduc_lines(vertical, 10)
lines = horizontal + vertical
for l in lines:
    cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)


# Var4: Works only if lines are exactly horizontal (l1[1] == l1[3]) and vertical (l2[0] == l2[2]). For our case is OK!

def intersect4(l1, l2):
    #    l1 - horizontal, l2 vertical, else swap
    if l2[1] == l2[3]:
        l1, l2 = l2, l1
    if (l1[0] <= l2[0] <= l1[2]) and (l2[1] <= l1[1] <= l2[3]):
        x, y = l2[0], l1[1]
        print(' --> intersect: ', x, y)
        return x, y
    else:
        print(' --> No intersection')
        return False


dots = []

for hor_line in horizontal:
    print(hor_line)
    for ver_line in vertical:
        print(ver_line, end='')
        inter = intersect4(hor_line, ver_line)
        if inter:
            dots.append(inter)
            cv2.circle(cdst, inter, 10, (255, 0, 0), 2)

dpi = plt.rcParams['figure.dpi']
height, width, depth = cdst.shape

# What size does the figure need to be in inches to fit the image?
figsize = width / float(dpi), height / float(dpi)

# Create a figure of the right size with one axes that takes up the full figure
fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0, 0, 1, 1])

# Hide spines, ticks, etc.
ax.axis('off')

# Display the image.
ax.imshow(cdst, cmap='gray')

plt.show()

# to have different filenames for different Canny parameters
fn2w = filename[:-4] + '-' + str(canny_params[0]) + '-' + str(canny_params[1]) + '.jpg'
cv2.imwrite(fn2w, cdst)

# TODO We need exactly correct mask of lines

# TODO Find locations (columns) of house numbers, names

# Find unique column_points and row_points
myset_x = set(x[0] for x in dots)
column_points = sorted(list(myset_x))
myset_y = set(x[1] for x in dots)
row_points = sorted(list(myset_y))

# crop image on intersection points, create pictures
for i in range(0, len(column_points)-1):
    for j in range(1, len(row_points)-2):
        crop_img = cdstP[row_points[j]:row_points[j+1], column_points[i]:column_points[i+1]].copy()  # using copy instead of original photo
        f_name = filename[:-4] + '-crop-' + str(i).zfill(2) + '-' + str(j).zfill(2) + '.jpg'
        cv2.imwrite(f_name, crop_img)  # instead of creating images, we can write images to 3D array cdst.shape