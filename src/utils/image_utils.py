import os
from collections import namedtuple

import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import logging

log = logging.getLogger(__name__)

OUTPUT_DIR = '../images/output/'


class Image:
    data: np.ndarray = None
    file_name: str = None

    def __init__(self, data, file_name):
        self.data = data
        self.file_name = file_name

    def copy(self):
        return Image(self.data.copy(), self.file_name)


def binarize_image(img: Image) -> Image:
    blured1 = cv2.medianBlur(img.data, 1)
    blured2 = cv2.medianBlur(img.data, 17)

    divided = np.ma.divide(blured1, blured2).data
    normed = np.uint8(255 * divided / divided.max())
    th, threshed = cv2.threshold(normed, 100, 255, cv2.THRESH_OTSU)

    return Image(255 - threshed, img.file_name)


def align_image(img: Image) -> Image:
    angle = compute_image_skew(img.data)
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


def rotate_image(img: Image, angle):
    if angle == 0:
        return img

    image_center = tuple(np.array(img.data.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(img.data, rot_mat, img.data.shape[1::-1], flags=cv2.INTER_LINEAR)

    return Image(result, img.file_name)


def load_image(src: str) -> Image:
    file_name = os.path.basename(src)
    data = cv2.imread(src, cv2.IMREAD_GRAYSCALE)

    return Image(data, file_name)


def write_image(img: Image, file_name='image.png', destination=''):
    cv2.imwrite(os.path.join(destination, file_name), img)


def write_cells(img: Image, cell_position, file_name='cells.png', destination=''):
    img = img.copy()

    for rec in cell_position:
        cv2.rectangle(img.data, rec[0], rec[-1], (255, 255, 255), 2)

    cv2.imwrite(os.path.join(destination, file_name), img.data)


def write_points(img: Image, points, file_name='dots.png', destination=''):
    img = img.copy()
    for row in points:
        for point in row:
            cv2.circle(img.data, tuple(point), 10, (255, 0, 0), 2)

    cv2.imwrite(os.path.join(destination, file_name), img)


def write_lines(img: Image, lines, file_name='lines.png', destination=''):
    img = img.copy()

    for line in lines:
        cv2.line(img.data, (line[0], line[1]), (line[2], line[3]), (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imwrite(os.path.join(destination, file_name), img)


def canny(img: Image) -> Image:
    canny_params =[]
    v = np.median(img.data)
    sigma = 0.33
    # apply automatic Canny edge detection using the computed median
    canny_params.append(int(max(0, (1.0 - sigma) * v)))
    canny_params.append(int(min(255, (1.0 + sigma) * v)))
    canny_data = cv2.Canny(img, canny_params[0], canny_params[1], None, 3)  # 200 -> 300

    return Image(canny_data, img.file_name)


def scale_image(img: Image, scale: int) -> Image:
    height, width = img.data.shape
    new_height, new_width = int(width / scale), int(height / scale)

    data = cv2.resize(img.data, (new_height, new_width))
    log.debug(f'Scaled from: "{(height, width)}" to: "{(new_height, new_width)}"')

    return Image(data, img.file_name)