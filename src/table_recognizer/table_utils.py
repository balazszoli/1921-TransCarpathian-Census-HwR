from collections import OrderedDict

import cv2
import numpy as np

from src.utils.math_utils import line_equation_coefficients, line_intersection

import logging

from utils.image_utils import binarize_image, align_image, Image
from utils.log_utils import log_image, log_lines

log = logging.getLogger(__name__)


def find_lines(img):
    lines = cv2.HoughLinesP(img.data, 1, np.pi / 180, 50, None, 220, 10)
    lines = [[line[0][0], line[0][1], line[0][2], line[0][3]] for line in lines]

    return lines


def find_table_cells_position(img):
    lines = find_lines(img)

    log_lines(img, lines)

    horizontal, vertical = continue_lines(img, lines)
    return find_line_intersection_points(horizontal, vertical)


def continue_lines(img: Image, lines: list):
    """
    Extract all horizontal and vertical lines from image
    and continue them to image border
    >>> continue_lines(Image(np.array([[[10, 60, 10, 30]]], 'file_name.png'), [])
    # [[],[[50, 0, 50, 100]]]

    @param img: image
    @param lines: a list of lines

    @return: return vertical and horizontal lines lists
    """
    img_height = img.data.shape[0]
    img_width = img.data.shape[1]

    horizontal = dict()
    vertical = dict()
    for line in lines:
        angel = np.arctan2(line[3] - line[1], line[2] - line[0]) * 180. / np.pi

        if -91 < angel < -89:
            vertical[line[0]] = [line[0], 0, line[2], img_height]
        elif angel == 0:
            horizontal[line[1]] = [0, line[1], img_width, line[3]]

    horizontal = OrderedDict(sorted(horizontal.items()))
    vertical = OrderedDict(sorted(vertical.items()))

    h = reduce_lines(horizontal, img_height)
    v = reduce_lines(vertical, img_width)
    return h, v


def reduce_lines(lines, border, threshold=10):
    if not lines:
        return []

    buckets = []
    prev = next(iter(lines.keys()))
    current_bucket = []

    for key in lines.keys():
        # Check if line is close to image border
        # than it would be the image border line which is not the table part
        # so skip them
        if key - 30 <= 0 or key + 30 > border:
            prev = key
            continue

        if key > threshold + prev:
            buckets.append(current_bucket)
            current_bucket = []

        current_bucket.append(key)
        prev = key

    if len(current_bucket) > 0:
        buckets.append(current_bucket)

    result = []

    for bucket in buckets:
        if len(bucket) == 0:
            continue
        mean = np.mean(bucket)
        result.append(lines[min(bucket, key=lambda x: abs(x - mean))])

    return result


def find_line_intersection_points(horizontal, vertical):
    dots = []

    # Find intersection of horizontal and vertical lines and plot them on image (just for observation)
    for hor_line in horizontal:
        L1 = line_equation_coefficients(hor_line[:2], hor_line[2:])
        row = []
        for ver_line in vertical:

            L2 = line_equation_coefficients(ver_line[:2], ver_line[2:])
            inter = line_intersection(L1, L2)
            if inter:
                row.append(inter)

        dots.append(row)

    return dots


def complete_table_from_template(template_table, actual_table):
    """
    While recognizing image we can lose some columns or rows
    Here we should use template table to complete actual table
    """

    template_table_top_left_angel = template_table[0][0]
    top_left_angel = actual_table[0][0]

    # This is the simples solution to adjust table to template
    # For first attempt would be OK :)
    # translation is a geometric transformation that moves every point of a figure or a space
    # by the same distance in a given direction.
    a = top_left_angel[0] - template_table_top_left_angel[0]
    b = top_left_angel[1] - template_table_top_left_angel[1]

    return [[(a + point[0], b + point[1]) for point in row] for row in template_table]
