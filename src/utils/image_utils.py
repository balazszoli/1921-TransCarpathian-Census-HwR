import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

OUTPUT_DIR = '../images/output/'

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


def write_cells(img, file_name, cell_position):
    for rec in cell_position:
        cv2.rectangle(img, rec[0], rec[-1], (255, 255, 255), 2)

    cv2.imwrite(OUTPUT_DIR + file_name + "_cells.jpg", img.copy())


def write_dots(img, positions):
    for pos in positions:
        cv2.circle(img, tuple(pos), 10, (255, 0, 0), 2)

    cv2.imwrite(OUTPUT_DIR + "dots.jpg", img.copy())


def write_lines(img, lines):
    for l in lines:
        l = l[0]
        cv2.line(img, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imwrite(OUTPUT_DIR + "lines.jpg", img.copy())
