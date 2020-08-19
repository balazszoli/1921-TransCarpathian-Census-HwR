import os

import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import logging

log = logging.getLogger(__name__)

OUTPUT_DIR = '../images/output/'


def binarize_image(img):
    blured1 = cv2.medianBlur(img, 1)
    blured2 = cv2.medianBlur(img, 31)
    divided = np.ma.divide(blured1, blured2).data
    normed = np.uint8(255 * divided / divided.max())
    th, threshed = cv2.threshold(normed, 100, 255, cv2.THRESH_OTSU)

    return 255 - threshed


def align_table(img):
    angle = compute_image_skew(img)
    img = rotate_image(img, angle)

    log.debug(f'The image skew angle is {angle}')

    return img


def compute_image_skew(img, delta=1, limit=5):
    def find_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score

    angles = np.arange(-limit, limit + delta, delta)
    scores = []

    for angle in angles:
        hist, score = find_score(img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]

    return best_angle


def rotate_image(image, angle):
    if angle == 0:
        return image

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


def load_image(src):
    return cv2.imread(src, cv2.IMREAD_GRAYSCALE)


def write_image(img, file_name='image.png', destination=''):
    cv2.imwrite(os.path.join(destination, file_name), img)


def write_cells(img, cell_position, file_name='dots.png', destination=''):
    img = img.copy()

    for rec in cell_position:
        cv2.rectangle(img, rec[0], rec[-1], (255, 255, 255), 2)

    cv2.imwrite(os.path.join(destination, file_name), img)


def write_points(img, points, file_name='dots.png', destination=''):
    img = img.copy()
    for row in points:
        for point in row:
            cv2.circle(img, tuple(point), 10, (255, 0, 0), 2)

    cv2.imwrite(os.path.join(destination, file_name), img)


def write_lines(img, lines, file_name='lines.png', destination=''):
    img = img.copy()

    for line in lines:
        line = line[0]
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imwrite(os.path.join(destination, file_name), img)
